import torch, os
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from .utils import display_data_shape, get_named_dict, one_hot_target,\
    softmax, sparse_crossentropy, calc_iou_per_class, log_data, load_config,\
    summary_write_masks, load_npy, numpy_to_tensor, tensor_to_numpy, prepare_inputs
    
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return torch.reshape(x, (x.shape[0], -1))

class Reshape(nn.Module):
  def __init__(self, dims):
    super(Reshape, self).__init__()
    self.dims = dims

  def forward(self, x):
    return x.view(self.dims)

class LEO(nn.Module):
    """
    contains functions to perform latent embedding optimization
    """
    def __init__(self, config, mode="meta_train"):
        super(LEO, self).__init__()
        self.config = config
        self.mode = mode
        self.latent_size = config.hyperparameters.num_latents
        self.dense_input_shape = (28, 192, 256)
        self.encoder = self.encoder_block(14, 28, dropout=True)
        self.decoder = self.decoder_block(28, 28, 28, 28, dropout=False)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
              
    def encoder_block(self, in_channels, out_channels, dropout=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        
        layers.extend([Flatten(), nn.Linear(int(np.prod(self.dense_input_shape)), 2*self.latent_size)])
        
        return nn.Sequential(*layers)

    def decoder_block(self, conv_tr_in_channels, conv_tr_out_channels, \
            in_channels, out_channels, dropout=False):
        layers = [  nn.Linear(self.latent_size, int(np.prod(self.dense_input_shape))),
                    nn.ReLU(True),
                    Reshape((-1,) + self.dense_input_shape),
                    nn.ConvTranspose2d(in_channels=conv_tr_in_channels,
                                    out_channels=conv_tr_out_channels,
                                    kernel_size=4, stride=2,\
                                    padding=1),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,\
                        stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
    
        return nn.Sequential(*layers)

    def forward_encoder(self, inputs):
        o = self.encoder(inputs)
        latents, mean, logvar = self.sample_latents(o)
        self.latents, self.mean, self.logvar = latents, mean, logvar
        #logpz = self.log_normal_pdf(latents, 0., 0.)
        #logqz_x = self.log_normal_pdf(latents, mean, logvar)
        #kl_loss = torch.mean(logqz_x - logpz )
        kl_loss =  -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp() + 1, dim=1)
        kl_loss = torch.mean(kl_loss)
        return latents, kl_loss
        
    def forward_decoder(self, inputs, latents, targets):
        predicted_weights = self.decoder(latents)
        #average codes per class
        predicted_weights = torch.unsqueeze(torch.mean(predicted_weights, 0), 0)
        inner_loss, pred = self.calculate_inner_loss(inputs, targets, predicted_weights)
        return inner_loss, predicted_weights, pred

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = torch.log(2. * np.pi)
        return torch.sum(
            -.5 * ((sample - mean).pow(2) * torch.exp(-logvar) + logvar + log2pi),
            dim=raxis)
        
    def sample_latents(self, encoder_output):
        split_size = int(encoder_output.shape[1]//2)
        splits = [split_size, split_size]
        mean, logvar = torch.split(encoder_output, splits, dim=1)
        eps_dist = Normal(torch.zeros(mean.size()), torch.ones(logvar.size()))
        eps = eps_dist.sample().to(self.device)
        latents = eps * torch.exp(logvar * 0.5) + mean
        return latents, mean, logvar
       
    def seg_network(self, inputs, predicted_weights):
        channel_zero = inputs * predicted_weights[:, :14, :, :]
        channel_zero = torch.unsqueeze(torch.mean(channel_zero, dim=1), 1)
        channel_one = inputs * predicted_weights[:, 14:, :, :]
        channel_one = torch.unsqueeze(torch.mean(channel_one, dim=1), 1)
        pred =  torch.cat([channel_zero, channel_one], 1)
        return pred

    def calculate_inner_loss(self, inputs, targets, predicted_weights):
        pred = self.seg_network(inputs, predicted_weights)
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

    def finetuning_inner_loop(self, data, tr_loss, seg_weights, mode):
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
            tr_loss, _ = self.calculate_inner_loss(data.tr_data, data.tr_data_masks, seg_weights)
        if mode == "meta_train":
            val_loss, predictions = self.calculate_inner_loss(data.val_data, data.val_data_masks, seg_weights)
            mean_iou = calc_iou_per_class(predictions, data.val_data_masks)
        else:
            mean_ious = []
            val_losses = []
            val_img_paths = data.val_data
            val_mask_paths = data.val_data_masks
            for _img_path, _mask_path in tqdm(zip(val_img_paths, val_mask_paths)):
                input_embedding = prepare_inputs(torch.unsqueeze(numpy_to_tensor(load_npy(_img_path)), 0))
                input_mask = prepare_inputs(torch.unsqueeze(numpy_to_tensor(load_npy(_mask_path)), 0))
                val_loss, prediction = self.calculate_inner_loss(input_embedding, input_mask, seg_weights)
                mean_iou = calc_iou_per_class(prediction, input_mask)
                mean_ious.append(mean_iou)
                val_losses.append(tensor_to_numpy(val_loss))
            mean_iou = np.mean(mean_ious)
            val_loss = np.mean(val_losses)
        return val_loss, mean_iou

    def forward(self, tr_data, tr_data_masks, val_data, val_masks):
        metadata = (tr_data, tr_data_masks, val_data, val_masks, "") 
        data_dict = get_named_dict(metadata, 0)
        latents, kl_loss = self.forward_encoder(data_dict.tr_data)
        inner_loss, _, _ = self.forward_decoder(data_dict.tr_data, latents, data_dict.tr_data_masks)
        #val_loss = self.finetuning_inner_loop(data_dict, tr_loss, adapted_seg_weights)
        #kl_loss = kl_loss * self.config.hyperparameters.kl_weight
        return inner_loss + kl_loss


    def compute_loss(self, metadata, train_stats, mode):
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
        if train_stats.episode % self.config.display_stats_interval == 1:
            display_data_shape(metadata, mode)
        classes = metadata[-1]
        total_val_loss = []
        kl_losses = []
        mean_iou_dict = {} 
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            latents, kl_loss = self.forward_encoder(data_dict.tr_data)

            tr_loss, adapted_seg_weights = self.leo_inner_loop(\
                            data_dict.tr_data, latents, data_dict.tr_data_masks)

            val_loss, mean_iou = self.finetuning_inner_loop(data_dict, tr_loss,\
                                                            adapted_seg_weights, mode)
            mean_iou_dict[classes[batch]] = mean_iou
            total_val_loss.append(val_loss)
            kl_loss = kl_loss * self.config.hyperparameters.kl_weight
            kl_losses.append(kl_loss)
            
        total_val_loss = sum(total_val_loss)/len(total_val_loss)
        total_kl_loss = sum(kl_losses)/len(kl_losses)
        total_loss = total_val_loss + total_kl_loss
        stats_data = {
            "mode": mode,
            "kl_loss": total_kl_loss,
            "total_val_loss":total_val_loss,
            "mean_iou_dict":mean_iou_dict
        }
        train_stats.update_stats(**stats_data)
        return total_loss, train_stats

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
        'total_val_loss': stats.total_val_loss,
        'mean_iou_dict': stats.mean_iou_dict 
        
    }

    experiment = config.experiment
    model_root = os.path.join(os.path.dirname(__file__), config.data_path, "models")
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
    model_dir  = os.path.join(os.path.dirname(__file__), config.data_path, "models", "experiment_{}"\
                 .format(experiment.number))
    
    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    #if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
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
    mean_iou_dict = checkpoint['mean_iou_dict']

    stats = {
        "mode": mode,
        "episode": episode,
        "kl_loss": kl_loss,
        "total_val_loss": total_val_loss,
        "mean_iou_dict": mean_iou_dict
        }

    return leo, optimizer, stats

