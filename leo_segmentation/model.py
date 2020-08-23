# Architecture definition
# Computational graph creation
import torch, os
from torch import nn
import numpy as np
from utils import display_data_shape, get_named_dict, one_hot_target,\
    softmax, sparse_crossentropy, calc_iou_per_class, log_data, load_config
    

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
        self.enc1 = _EncoderBlock(14, 32)
        self.enc2 = _EncoderBlock(32, 32, dropout=True) 
        self.dec2 = _DecoderBlock(32, 32, 32, 32)
        self.dec1 = _DecoderBlock(32, 32, 32, 28)
        
    def forward_encoder(self, inputs):
        o = self.enc1(inputs)
        o = self.enc2(o)
        return o

    def forward_decoder(self, inputs, latents, targets):
        o = self.dec2(latents)
        predicted_weights = self.dec1(o)
        #average codes per class
        predicted_weights = torch.unsqueeze(torch.mean(predicted_weights, 0), 0)
        inner_loss, pred = self.calculate_inner_loss(inputs, targets, predicted_weights)
        return inner_loss, predicted_weights, pred
       

    def calculate_inner_loss(self, inputs, targets, predicted_weights):
        channel_zero = inputs * predicted_weights[:, :14, :, :]
        channel_zero = torch.unsqueeze(torch.mean(channel_zero, dim=1), 1)
        channel_one = inputs * predicted_weights[:, 14:, :, :]/5.0
        channel_one = torch.unsqueeze(torch.mean(channel_one, dim=1), 1)
        pred =  torch.cat([channel_zero, channel_one], 1)
        hot_targets = one_hot_target(targets)
        hot_targets.requires_grad = True
        pred = torch.clamp(pred, -10, 3.0)
        pred = softmax(pred)
        loss = sparse_crossentropy(hot_targets, pred)
        return loss, pred
        

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
        initial_latents = latents.clone()
        tr_loss, _ , _ = self.forward_decoder(inputs, latents, target)
        
        for _ in range(self.config.hyperparameters.num_adaptation_steps):
            latents_grad = torch.autograd.grad(tr_loss, [latents], create_graph=False)[0]
            with torch.no_grad():
                latents -= inner_lr * latents_grad
            tr_loss, segmentation_weights, _ = self.forward_decoder(inputs, latents, target)
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
            seg_weights_grad = torch.autograd.grad(tr_loss, [seg_weights])[0]
            with torch.no_grad():
                seg_weights -= finetuning_lr * seg_weights_grad
                #self.scale -= finetuning_lr * scale_grad
            tr_loss, _ = self.calculate_inner_loss(data.tr_data, data.tr_data_masks, seg_weights)
        val_loss, _ = self.calculate_inner_loss(data.val_data, data.val_data_masks, seg_weights)
        return val_loss

    def compute_loss(self, metadata, train_stats, config, mode="meta_train"):
        """
        Computes the  outer loop loss

        Args:
            model (object) : leo model
            meta_dataloader (object): Dataloader 
            train_stats: (object): train stats object
            config (dict): config
            mode (str): meta_train, meta_val or meta_test
        Returns:
            (tuple) total_val_loss (list), train_stats
        """
        num_tasks = len(metadata[0])
        if train_stats.episode % config.display_stats_interval == 1:
            display_data_shape(metadata)
        total_val_loss = []
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            latents = self.forward_encoder(data_dict.tr_data)
            tr_loss, adapted_seg_weights = self.leo_inner_loop(\
                            data_dict.tr_data, latents, data_dict.tr_data_masks)
            val_loss = self.finetuning_inner_loop(data_dict, tr_loss, adapted_seg_weights)
            total_val_loss.append(val_loss)
            
        total_val_loss = sum(total_val_loss)/len(total_val_loss)
        stats_data = {
            "mode": mode,
            "kl_loss": 0,
            "total_val_loss":total_val_loss
        }
        train_stats.update_stats(**stats_data)
        return total_val_loss, train_stats


    def evaluate_val_data(self, metadata, classes, train_stats):
        num_tasks = len(metadata[0])
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            latents = self.forward_encoder(data_dict.val_data)
            _, _, predictions = self.forward_decoder(data_dict.val_data, latents, data_dict.val_data_masks)
            iou = calc_iou_per_class(predictions, data_dict.val_data_masks)
            print(f"Class: {classes[batch]}, Episode: {train_stats.episode}, Val IOU: {iou}")


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
        'kl_loss': stats.kl_loss,
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

def load_model(config):

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
    model_dir  = os.path.join(config.data_path, "models", "experiment_{}"\
                 .format(experiment.number))
    
    checkpoints = os.listdir(model_dir)
    checkpoints.pop()
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    #if experiment.episode == -1, load latest checkpoint
    episode = experiment.episode if experiment.episode == -1 else max_cp
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg =  f"\n*********** checkpoint {episode} was loaded **************" 
    log_data(msg, log_filename)
    
    leo = LEO(config)
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
    kl_loss = checkpoint['kl_loss']

    stats = {
        "mode": mode,
        "episode": episode,
        "kl_loss": kl_loss,
        "total_val_loss": total_val_loss
        }

    return leo, optimizer, stats