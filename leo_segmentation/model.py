import os
import torch
import gc
import numpy as np
from torch import nn
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch.nn import functional as F
from tqdm import tqdm
from .utils import display_data_shape, get_named_dict, calc_iou_per_class,\
    log_data, load_config, list_to_tensor, numpy_to_tensor, tensor_to_numpy,\
    prepare_inputs


class EncoderBlock(nn.Module):
    """ Encoder with pretrained backbone """
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList(list(models.mobilenet_v2(pretrained=True)
                                    .features))
    
    def forward(self, x):
        features = []
        output_layers = [1, 3, 6, 13]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_layers:
                features.append(x)
        features.append(x)
        return features


def decoder_block(conv_in_size, conv_out_size):
    """ Sequentical group formimg a decoder block """
    layers = [
              nn.Conv2d(conv_in_size, conv_out_size,
                        kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.Dropout(hyp.dropout_rate),
              nn.BatchNorm2d(conv_out_size),
              nn.Conv2d(conv_out_size, conv_out_size,
                        kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(conv_out_size, conv_out_size,
                                 kernel_size=4, stride=2, padding=1)
             ]
    conv_block = nn.Sequential(*layers)
    return conv_block


class DecoderBlock(nn.Module):
    """
    Leo Decoder
    """
    def __init__(self, encoder_outputs):
        super(DecoderBlock, self).__init__()
        self.conv1 = decoder_block(encoder_outputs[-1].shape[1],
                                   hyp.base_num_covs*1)
        self.conv2 = decoder_block(encoder_outputs[-2].shape[1] + hyp.base_num_covs*1, 
                                   hyp.base_num_covs*2)
        self.conv3 = decoder_block(encoder_outputs[-3].shape[1] + hyp.base_num_covs*2,
                                   hyp.base_num_covs*3)
        self.conv4 = decoder_block(encoder_outputs[-4].shape[1] + hyp.base_num_covs*3,
                                   hyp.base_num_covs*4)
        self.up_final = nn.ConvTranspose2d(encoder_outputs[-5].shape[1] + hyp.base_num_covs*4,
                                   hyp.base_num_covs*5, kernel_size=4, stride=2, padding=1)
        
    def forward(self, encoder_outputs):
        o = self.conv1(encoder_outputs[-1])
        o = torch.cat([o, encoder_outputs[-2]], dim=1)
        o = self.conv2(o)
        o = torch.cat([o, encoder_outputs[-3]], dim=1)
        o = self.conv3(o)
        o = torch.cat([o, encoder_outputs[-4]], dim=1)
        o = self.conv4(o)
        o = torch.cat([o, encoder_outputs[-5]], dim=1)
        o = self.up_final(o)
        return o


class LEO(nn.Module):
    """
    contains functions to perform latent embedding optimization
    """
    def __init__(self, mode="meta_train"):
        super(LEO, self).__init__()
        self.mode = mode
        img_dims = config.data_params.img_dims
        self.img_dims = (img_dims.channels, img_dims.height, img_dims.width)
        self.encoder = EncoderBlock()
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and config.use_gpu else "cpu")
        seg_network = nn.Conv2d(hyp.base_num_covs*5 + 3, 2, kernel_size=3, stride=1, padding=1)
        self.seg_weight = seg_network.weight.detach().to(self.device)
        self.seg_weight.requires_grad = True
        self.loss_fn = CrossEntropyLoss()
        self.optimizer_seg_network = torch.optim.Adam(
            [self.seg_weight], lr=hyp.outer_loop_lr)

    def freeze_encoder(self):
        """ Freeze encoder weights """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward_encoder(self, x):
        """ Performs forward pass through the encoder """
        encoder_outputs = self.encoder(x)
        if not encoder_outputs[-1].requires_grad:
            encoder_outputs[-1].requires_grad = True
        return encoder_outputs

    def forward_decoder(self, encoder_outputs):
        """Performs forward pass through the decoder"""
        output = self.decoder(encoder_outputs)
        return output

    def forward_segnetwork(self, decoder_out, x, weight):
        """  Receives features from the decoder
             Concats the features with input image
             Convolution layer acts on the concatenated input
            Args:
                decoder_out (torch.Tensor): decoder output features
                x (torch.Tensor): input images
                weight(tf.tensor): kernels for the segmentation network
            Returns:
                pred(tf.tensor): predicted logits
        """
        o = torch.cat([decoder_out, x], dim=1)
        pred = F.conv2d(o, weight, padding=1)
        return pred

    def forward(self, x, latents=None, weight=None):
        """ Performs a forward pass through the entire network
            - The Autoencoder generates features using the inputs
            - Features are concatenated with the inputs
            - The concatenated features are segmented
            Args:
                x (torch.Tensor): input image
                latents(torch.Tensor): output of the bottleneck
            Returns:
                latents(torch.Tensor): output of the bottleneck
                features(torch.Tensor): output of the decoder
                pred(torch.Tensor): predicted logits
                weight(torch.Tensor): segmentation weights
        """
        encoder_outputs = self.forward_encoder(x)
        if latents is not None:
            encoder_outputs = encoder_outputs[:4] + [latents]
        else:
            latents = encoder_outputs[-1]

        if weight is not None:
            seg_weight = weight
        else:
            seg_weight = self.seg_weight
        features = self.forward_decoder(encoder_outputs)
        pred = self.forward_segnetwork(features, x, seg_weight)
        return latents, features, pred
    
    def leo_inner_loop(self, x, y):
        """ Performs innerloop optimization
            - It updates the latents taking gradients wrt the training loss
            - It generates better features after the latents are updated

            Args:
                x(torch.Tensor): input training image
                y(torch.Tensor): input training mask

            Returns:
                seg_weight_grad(torch.Tensor): The last gradient of the
                    training loss wrt to the segmenation weights
                features(torch.Tensor): The last generated features
        """    
        inner_lr = hyp.inner_loop_lr
        latents, _, pred = self.forward(x)
        tr_loss = self.loss_fn(pred, y.long())
        for _ in range(hyp.num_adaptation_steps):
            latents_grad = torch.autograd.grad(tr_loss, [latents],
                                               create_graph=False)[0]
            with torch.no_grad():
                latents -= inner_lr * latents_grad
            latents, features, pred = self.forward(x, latents)
            tr_loss = self.loss_fn(pred, y.long())
        seg_weight_grad = torch.autograd.grad(tr_loss, [self.seg_weight],
                                              create_graph=False)[0]
        return seg_weight_grad, features

    def finetuning_inner_loop(self, data_dict, tr_features, seg_weight_grad,
                              transformers, mode):
        """ Finetunes the segmenation weights/kernels by performing MAML
            Args:
                data_dict (dict): contains tr_imgs, tr_masks, val_imgs, val_masks
                tr_features (torch.Tensor): tensor containing decoder features
                segmentation_grad (torch.Tensor): gradients of the training
                                                loss to the segmenation weights
            Returns:
                val_loss (torch.Tensor): validation loss
                seg_weight_grad (torch.Tensor): gradient of validation loss
                                                wrt segmentation weights
                decoder_grads (torch.Tensor): gradient of validation loss
                                                wrt decoder weights
                transformers(tuple): tuple of image and mask transformers
                weight (torch.Tensor): segmentation weights
        """
        img_transformer, mask_transformer = transformers
        weight = self.seg_weight - hyp.finetuning_lr * seg_weight_grad
        for _ in range(hyp.num_finetuning_steps - 1):
            pred = self.forward_segnetwork(tr_features, data_dict.tr_imgs, weight)
            tr_loss = self.loss_fn(pred, data_dict.tr_masks.long())
            seg_weight_grad = torch.autograd.grad(tr_loss, [weight],
                                                  create_graph=False)[0]
            weight -= hyp.finetuning_lr * seg_weight_grad

        if mode == "meta_train":
            _, _, prediction = self.forward(data_dict.val_imgs, weight=weight)
            val_loss = self.loss_fn(prediction, data_dict.val_masks.long())
            grad_output = torch.autograd.grad(val_loss,
                [weight] + list(self.decoder.parameters()), create_graph=False)
            seg_weight_grad, decoder_grads = grad_output[0], grad_output[1:]
            mean_iou = calc_iou_per_class(prediction, data_dict.val_masks)
            return val_loss, seg_weight_grad, decoder_grads, mean_iou, weight
        else:
            with torch.no_grad():
                mean_ious = []
                val_losses = []
                val_img_paths = data_dict.val_imgs
                val_mask_paths = data_dict.val_masks
                for _img_path, _mask_path in tqdm(zip(val_img_paths, val_mask_paths)):
                    input_img = prepare_inputs(numpy_to_tensor(list_to_tensor(_img_path, img_transformer)))
                    input_mask = numpy_to_tensor(list_to_tensor(_mask_path, mask_transformer))
                    _, _, prediction = self.forward(input_img, weight=weight)
                    val_loss = self.loss_fn(prediction, input_mask.long()).item()
                    mean_iou = calc_iou_per_class(prediction, input_mask)
                    mean_ious.append(mean_iou)
                    val_losses.append(val_loss)
                mean_iou = np.mean(mean_ious)
                val_loss = np.mean(val_losses)
            return val_loss, None, None, mean_iou, weight
        
    def compute_loss(self, metadata, train_stats, transformers, mode="meta_train"):
        """ Performs meta optimization across tasks
            returns the meta validation loss across tasks
            Args:
                metadata(dict): dictionary containing training data
                train_stats(object): object that stores training statistics
                transformers(tuple): tuple of image and mask transformers
                mode(str): meta_train, meta_val or meta_test
            Returns:
                total_val_loss(float32): meta-validation loss
                train_stats(object): object that stores training statistics
        """
        num_tasks = len(metadata[0])
        # initialize decoder on the first episode
        if train_stats.episode == 1:
            data = get_named_dict(metadata, 0)
            encoder_output = self.forward_encoder(data.tr_imgs)
            self.decoder = DecoderBlock(encoder_output).to(self.device)
            self.optimizer_decoder = torch.optim.Adam(
              self.decoder.parameters(), lr=hyp.outer_loop_lr)

        if train_stats.episode % config.display_stats_interval == 1:
            display_data_shape(metadata)
        classes = metadata[4]
        total_val_loss = []
        mean_iou_dict = {}
        total_grads = None
        for batch in range(num_tasks):
            data = get_named_dict(metadata, batch)
            seg_weight_grad, features = self.leo_inner_loop(data.tr_imgs, data.tr_masks)
            val_loss, seg_weight_grad, decoder_grads, mean_iou, _ = \
                self.finetuning_inner_loop(data, features, seg_weight_grad,
                                           transformers, mode)
            if mode == "meta_train":
                decoder_grads = [grad/num_tasks for grad in decoder_grads]
                if total_grads is None:
                    total_grads = decoder_grads
                    seg_weight_grad = seg_weight_grad/num_tasks
                else:
                    total_grads = [total_grads[i] + decoder_grads[i]\
                                   for i in range(len(decoder_grads))]
                    seg_weight_grad += seg_weight_grad/num_tasks
            mean_iou_dict[classes[batch]] = mean_iou
            total_val_loss.append(val_loss)

        if mode == "meta_train":
            self.optimizer_decoder.zero_grad()
            self.optimizer_seg_network.zero_grad()
            
            for i, params in enumerate(self.decoder.parameters()):
                params.grad = total_grads[i]
            self.seg_weight.grad = seg_weight_grad
            self.optimizer_decoder.step()
            self.optimizer_seg_network.step()
        total_val_loss = float(sum(total_val_loss)/len(total_val_loss))
        stats_data = {
            "mode": mode,
            "total_val_loss": total_val_loss,
            "mean_iou_dict": mean_iou_dict
        }
        train_stats.update_stats(**stats_data)
        return total_val_loss, train_stats


def save_model(model, optimizer, config, stats):
    """
    Save the model while training based on check point interval
    
    if episode number is not -1 then a prompt to delete checkpoints occur if 
    checkpoints for that episode number exits.
    This only occurs if the prompt_deletion flag in the experiment dictionary
    is true else checkpoints that already exists are automatically deleted

    Args:
        model - trained model
        optimizer - optimized weights
        config - global config
        stats - dictionary containing stats for the current episode
    
    Returns:
    """
    data_to_save = {
        'mode': stats.mode,
        'episode': stats.episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_val_loss': stats.total_val_loss
    }

    experiment = config.experiment
    model_root = os.path.join(config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))

    checkpoint_path = os.path.join(model_dir, f"checkpoint_{stats.episode}.pth.tar")
    if not os.path.exists(checkpoint_path):
        torch.save(data_to_save, checkpoint_path)
    else:
        trials = 0
        while trials < 3:
            if experiment.prompt_deletion:
                print(f"Are you sure you want to delete checkpoint: {stats.episode}")
                print(f"Type Yes or y to confirm deletion else No or n")
                user_input = input()
            else:
                user_input = "Yes"
            positive_options = ["Yes", "y", "yes"]
            negative_options = ["No", "n", "no"]
            if user_input in positive_options:
                # delete checkpoint
                os.remove(checkpoint_path)
                torch.save(data_to_save, checkpoint_path)
                log_filename = os.path.join(model_dir, "model_log.txt")
                msg = msg = f"\n*********** checkpoint {stats.episode} was deleted **************"
                log_data(msg, log_filename)
                break

            elif user_input in negative_options:
                raise ValueError("Supply the correct episode number to start experiment")
            else:
                trials += 1
                print("Wrong Value Supplied")
                print(f"You have {3 - trials} left")
                if trials == 3:
                    raise ValueError("Supply the correct answer to the question")


def load_model():

    """
    Loads the model
    Args:
        config - global config
        **************************************************
        Note: The episode key in the experiment dict
        implies the checkpoint that should be loaded
        when the model resumes training. If episode is
        -1, then the latest model is loaded else it loads
        the checkpoint at the supplied episode
        *************************************************
    Returns:
        leo :loaded model that was saved
        optimizer: loaded weights of optimizer
        stats: stats for the last saved model
    """
    experiment = config.experiment
    model_dir = os.path.join(config.data_path, "models", "experiment_{}"
                 .format(experiment.number))
    
    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    #if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg = f"\n*********** checkpoint {episode} was loaded **************" 
    log_data(msg, log_filename)
    
    leo = LEO()
    optimizer = torch.optim.Adam(leo.parameters(), lr=hyp.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
  
    stats = {
        "mode": mode,
        "episode": episode,
        "total_val_loss": total_val_loss
        }

    return leo, optimizer, stats


config = load_config()
hyp = config.hyperparameters
