# Architecture definition
# Computational graph creation
import torch
from torch import nn
import numpy as np
from torch.distributions import Normal

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, \
                kernel_size=3, stride=2, padding=1, \
                dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,\
                stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlock(nn.Module):
    def __init__(self, conv_tr_in_channels, conv_tr_out_channels, \
                       in_channels, out_channels, kernel_size=4,dropout=False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=conv_tr_in_channels,
                                    out_channels=conv_tr_out_channels,
                                    kernel_size=kernel_size, stride=2,\
                                    padding=1),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,\
                        stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False))

    def forward(self, x):
        return self.decode(x)

class LEO(nn.Module):
    """
    contains functions to perform latent embedding optimization
    """
    def __init__(self, config, mode="meta_train"):
        super(LEO, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.mode = mode
        self.loss = nn.CrossEntropyLoss()
        self.enc1 = _EncoderBlock(14, 32).to(self.device) 
        self.enc2 = _EncoderBlock(32, 32, dropout=True).to(self.device) 
        self.dec2 = _DecoderBlock(32, 32, 32, 32).to(self.device) 
        self.dec1 = _DecoderBlock(32, 32, 32, 28).to(self.device)  
        
    def forward_encoder(self, inputs):
        o = self.enc1(inputs)
        o = self.enc2(o)
        return o

    def forward_decoder(self, inputs, latents, targets):
        o = self.dec2(latents)
        predicted_weights = self.dec1(o)
        #average codes per class
        predicted_weights = torch.unsqueeze(torch.mean(predicted_weights, 0), 0)
        inner_loss = self.calculate_inner_loss(inputs, targets, predicted_weights)
        return inner_loss, predicted_weights
       

    def calculate_inner_loss(self, inputs, targets, predicted_weights):
        channel_zero_bias = 0.0
        channel_one_bias = 0.0
        scale = 10.0
        channel_zero = inputs * predicted_weights[:, :14, :, :]/scale + channel_zero_bias
        channel_zero = torch.unsqueeze(torch.sum(channel_zero, dim=1), 1)
        channel_one = inputs * predicted_weights[:, 14:, :, :]/scale + channel_one_bias
        channel_one = torch.unsqueeze(torch.sum(channel_one, dim=1), 1)
        pred =  torch.cat([channel_zero, channel_one], 1)
        hot_targets = one_hot_target(targets).to(self.device)
        hot_targets.requires_grad = True
        pred = softmax(pred)
        loss = sparse_crossentropy(hot_targets, pred)
        return loss
        

    def leo_inner_loop(self, inputs, latents, target):
        """
        This function does "latent code optimization" that is back propagation  until latent codes and
        updating the latent weights
        Args:
            data (dict) : contains tr_data, tr_data_masks, val_data, val_data_masks
            latents (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W)
        Returns:
            tr_loss : computed as crossentropyloss (groundtruth--> tr_data_mask, prediction--> einsum(tr_data, segmentation_weights))
            segmentation_weights : shape(num_classes, num_eg_per_class, channels, H, W)
        """
        inner_lr = self.config.hyperparameters.inner_loop_lr
        latents.retain_grad()
        initial_latents = latents.clone()
        #weights_shape = (1,) + inputs.shape[1:]
        #self.segnet = Segmodel(weights_shape).to(self.device) 
        tr_loss, _  = self.forward_decoder(inputs, latents, target)
        print(tr_loss)
        
        for _ in range(self.config.hyperparameters.num_adaptation_steps):
            grad = torch.autograd.grad(tr_loss, latents, create_graph=True)[0]
            with torch.no_grad():
                latents -= inner_lr * grad
                #latents.grad.zero_()
            tr_loss, segmentation_weights = self.forward_decoder(inputs, latents, target)
        return tr_loss, segmentation_weights

    def finetuning_inner_loop(self, data, tr_loss, seg_weights):
        """
        This function does "segmentation_weights optimization"
        Args:
            data (dict) : contains tr_data, tr_data_masks, val_data, val_data_masks
            leo_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> tr_data_mask, prediction--> einsum(tr_data, segmentation_weights))
           segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        Returns:
            val_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> val_data_mask, prediction--> einsum(val_data, segmentation_weights))
        """
        finetuning_lr = self.config.hyperparameters.finetuning_lr
        for _ in range( self.config.hyperparameters.num_finetuning_steps):
            grad_ = torch.autograd.grad(tr_loss, seg_weights)[0]
            with torch.no_grad():
                seg_weights -= finetuning_lr * grad_
            tr_loss = self.calculate_inner_loss(data.tr_data, data.tr_data_masks, seg_weights)
        val_loss = self.calculate_inner_loss(data.val_data, data.val_data_masks, seg_weights)
        print(f"Train loss {tr_loss}, val loss {val_loss}")
        return val_loss

    def grads_and_vars(self, metatrain_loss):
        """
        this function retrieves current gradient values for (Encoder, Decoder, Relation Network) LEO
        Args:
            metatrain_loss (tensor_oshape) : mean validation loss of LEO model for the entire data batches
        Returns:
            metatrain_variables (object) : parameter list (Encoder, Decoder, Relation Network)
            metatrain_gradients (tuple) : gradients of metatrain_variables w.r.t metatrain_loss
        """
        metatrain_variables = self.parameters()
        metatrain_gradients = torch.autograd.grad(metatrain_loss, metatrain_variables, retain_graph=True)
        #nan_loss_or_grad = torch.isnan(metatrain_loss)| torch.reduce([torch.reduce(torch.isnan(g))for g in metatrain_gradients])
        return metatrain_gradients, metatrain_variables

class Segmodel(nn.Module):
    def __init__(self, weight_shape):
        super(Segmodel, self).__init__()
        self.channel_zero_weight = nn.Parameter(torch.normal(torch.zeros(weight_shape), torch.ones(weight_shape)),\
                                            requires_grad=True)
        self.channel_one_weight = nn.Parameter(torch.normal(torch.zeros(weight_shape), torch.ones(weight_shape)),\
                                            requires_grad=True)
        #self.channel_one_bias = nn.Parameter(torch.normal(torch.zeros(weight_shape), torch.ones(weight_shape)),\
        #                                   requires_grad=True)
        #self.channel_two_bias = nn.Parameter(torch.normal(torch.zeros(weight_shape), torch.ones(weight_shape)),\
        #                                    requires_grad=True)
        #self.scale = nn.Parameter(torch.tensor(10.0, requires_grad=True))
        self.channel_one_bias = 1.0
        self.channel_two_bias = 1.0
        self.scale = 10.0


    def forward(self, x):
        channel_zero = x * self.channel_zero_weight/self.scale + self.channel_one_bias
        channel_zero = torch.unsqueeze(torch.sum(channel_zero, dim=1), 1)
        channel_one = x * self.channel_one_weight/self.scale + self.channel_two_bias
        channel_one = torch.unsqueeze(torch.sum(channel_one, dim=1), 1)
        return torch.cat([channel_zero, channel_one], 1)

def calc_iou_per_class(pred_x, targets):
    iou_per_class = []
    for i in range(len(pred_x)):
        pred = np.argmax(pred_x[i].cpu().detach().numpy(), 0).astype(int)
        target = targets[i].astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
        mean_iou_per_class = np.mean(iou_per_class)
    return mean_iou_per_class

def one_hot_target(mask, channel_dim=1):
    mask_inv = (~mask.type(torch.bool)).type(torch.float32)
    channel_zero = torch.unsqueeze(mask_inv, channel_dim)
    channel_one = torch.unsqueeze(mask, channel_dim)
    return torch.cat((channel_zero, channel_one), axis=channel_dim)

def softmax(py_tensor, channel_dim=1):
    py_tensor = torch.exp(py_tensor)
    return  py_tensor/torch.unsqueeze(torch.sum(py_tensor, dim=channel_dim), channel_dim)

def sparse_crossentropy(target, pred,  channel_dim=1, eps=1e-10):
    pred += eps
    loss = torch.sum(-1 * target * torch.log(pred), dim=channel_dim)
    return torch.mean(loss)