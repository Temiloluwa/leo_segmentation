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
        self.mode = mode
        self.loss = nn.CrossEntropyLoss()
        self.enc1 = _EncoderBlock(14, 32)
        self.enc2 = _EncoderBlock(32, 32, dropout=True)
        self.dec2 = _DecoderBlock(32, 32, 32, 32)
        self.dec1 = _DecoderBlock(32, 32, 32, 14) 
        self.decoder_loss = torch.nn.MSELoss(reduction="mean")
        self.seg_net_act = torch.nn.ReLu() 

    def forward_encoder(self, inputs):
        o = self.enc1(inputs)
        o = self.enc2(o)
        return o

    def forward_decoder(self, inputs, latents, target):
        o = self.dec2(latents)
        predicted_weights = self.dec1(o)
        inner_loss = self.calculate_inner_loss(inputs, predicted_weights, targets)
        return inner_loss, predicted_weights
       

    def calculate_inner_loss(self, inputs, weights, targets):
        num_channels, _, _, _ = weights.shape
        out_channel1 = torch.sum(inputs * weights[:int(num_channels/2), :, :, :], dim=0)
        out_channel2 = torch.sum(inputs * weights[int(num_channels/2):, :, :, :], dim=0)
        output = torch.cat((out_channel1, out_channel2), 0)
        prediction = torch.argmax(output, dim=0)
        prediction = self.seg_net_act(prediction)
        return self.decoder_loss(prediction, targets)
        

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
        tr_loss, _  = self.forward_decoder(inputs, latents, target)
        print(tr_loss)
        
        for _ in range(self.config.hyperparameters.num_adaptation_steps):
            tr_loss.backward()
            grad = latents.grad
            with torch.no_grad():
                latents -= inner_lr * grad
                latents.grad.zero()
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
            grad_ = torch.autograd.grad(tr_loss, seg_weights)
            with torch.no_grad():
                seg_weights -= finetuning_lr * grad_
            tr_loss = self.calculate_inner_loss(data.tr_data, seg_weights, data.tr_data_masks)
        val_loss = self.calculate_inner_loss(data.val_data, seg_weights, data.val_data_masks)

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
