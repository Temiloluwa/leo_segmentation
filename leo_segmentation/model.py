# Architecture definition
# Computational graph creation
import torch, os, numpy as np
from torch import nn
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from  torch.nn import functional as F
from utils import display_data_shape, get_named_dict, calc_iou_per_class,\
    log_data, load_config, summary_write_masks
    
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
        self.decoder = self.decoder_block(14, 14, 3, dropout=False)
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.loss_fn = CrossEntropyLoss()
              
    def encoder_block(self, in_channels, out_channels, dropout=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        
        layers.extend([Flatten(), nn.Linear(int(np.prod(self.dense_input_shape)), 2*self.latent_size)])
        
        return nn.Sequential(*layers)

    def decoder_block(self, input_channels, output_channels, kernel_size, dropout=False):
        output_size = input_channels * kernel_size**2 
        output_shape = (input_channels, kernel_size, kernel_size)
        layers = [  nn.Linear(self.latent_size, output_size),
                    nn.ReLU(True),
                    Reshape((-1,) + output_shape),
                    nn.Conv2d(input_channels, output_channels*2, kernel_size,\
                        stride=1, padding=1),
                    nn.BatchNorm2d(output_channels*2),
                    nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
    
        return nn.Sequential(*layers)

    def forward_encoder(self, inputs):
        o = self.encoder(inputs)
        latents, mean, logvar = self.sample_latents(o)
        self.latents, self.mean, self.logvar = latents, mean, logvar
        kl_loss =  -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp() + 1, dim=1)
        kl_loss = torch.mean(kl_loss)
        return latents, kl_loss
        
    def forward_decoder(self, inputs, latents, targets):
        kernels = self.decoder(latents)
        kernels = self.reshape_output(kernels)
        inner_loss, pred = self.calculate_inner_loss(inputs, targets, kernels)
        return inner_loss, kernels, pred

    def reshape_output(self, decoder_output):
        decoder_output = torch.mean(decoder_output, 0)
        channels, ks , ks = decoder_output.shape
        kernels = torch.reshape(decoder_output, (2, int(channels/2), ks , ks))
        return kernels
        
    def sample_latents(self, encoder_output):
        split_size = int(encoder_output.shape[1]//2)
        splits = [split_size, split_size]
        mean, logvar = torch.split(encoder_output, splits, dim=1)
        eps_dist = Normal(torch.zeros(mean.size()), torch.ones(logvar.size()))
        eps = eps_dist.sample().to(self.device)
        latents = eps * torch.exp(logvar * 0.5) + mean
        return latents, mean, logvar
       
    def seg_network(self, inputs, kernels):
        return F.conv2d(inputs, kernels, stride=1 ,padding=1)

    def calculate_inner_loss(self, inputs, targets, kernels):
        pred = self.seg_network(inputs, kernels)
        loss = self.loss_fn(pred, targets.long())
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
            tr_loss, kernels, _ = self.forward_decoder(inputs, latents, target)
        return tr_loss, kernels

    def finetuning_inner_loop(self, data, tr_loss, kernels):
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
            grad_kernels = torch.autograd.grad(tr_loss, [kernels])[0]
            with torch.no_grad():
                kernels -= finetuning_lr * grad_kernels
            tr_loss, _ = self.calculate_inner_loss(data.tr_data, data.tr_data_masks, kernels)
        val_loss, _ = self.calculate_inner_loss(data.val_data, data.val_data_masks, kernels)
        return val_loss

    def forward(self, tr_data, tr_data_masks, val_data, val_masks):
        metadata = (tr_data, tr_data_masks, val_data, val_masks, "") 
        data_dict = get_named_dict(metadata, 0)
        latents, kl_loss = self.forward_encoder(data_dict.tr_data)
        inner_loss, _, _ = self.forward_decoder(data_dict.tr_data, latents, data_dict.tr_data_masks)
        #val_loss = self.finetuning_inner_loop(data_dict, tr_loss, adapted_seg_weights)
        #kl_loss = kl_loss * self.config.hyperparameters.kl_weight
        return inner_loss + kl_loss


    def compute_loss(self, metadata, train_stats, mode="meta_train"):
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
            display_data_shape(metadata)
        total_val_loss = []
        kl_losses = []
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            latents, kl_loss = self.forward_encoder(data_dict.tr_data)

            tr_loss, adapted_kernels = self.leo_inner_loop(\
                            data_dict.tr_data, latents, data_dict.tr_data_masks)

            val_loss = self.finetuning_inner_loop(data_dict, tr_loss, adapted_kernels)
            total_val_loss.append(val_loss)
            kl_loss = kl_loss * self.config.hyperparameters.kl_weight
            kl_losses.append(kl_loss)

        total_val_loss = sum(total_val_loss)/len(total_val_loss)
        total_kl_loss = sum(kl_losses)/len(kl_losses)
        total_loss = total_val_loss + total_kl_loss
        stats_data = {
            "mode": mode,
            "kl_loss": total_kl_loss,
            "total_val_loss":total_val_loss
        }
        train_stats.update_stats(**stats_data)
        return total_loss, train_stats


    def evaluate_val_data(self, metadata, classes, train_stats, writer):
        log_msg = ""
        num_tasks = len(metadata[0])
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            latents, _ = self.forward_encoder(data_dict.val_data)
            _, _,  predictions = self.forward_decoder(data_dict.val_data, latents, data_dict.val_data_masks)
            iou = calc_iou_per_class(predictions, data_dict.val_data_masks)
            batch_msg = f"\nClass: {classes[batch]}, Episode: {train_stats.episode}, Val IOU: {iou}"
            print(batch_msg[1:])
            log_msg += batch_msg
            grid_title = f"pred_{train_stats.episode}_class_{classes[batch]}"
            summary_write_masks(predictions, writer, grid_title)
            grid_title = f"ground_truths_{train_stats.episode}_class_{classes[batch]}"
            summary_write_masks(data_dict.val_data_masks, writer, grid_title, ground_truth=True)
        log_filename = os.path.join(os.path.dirname(__file__), "data", "models",\
                         f"experiment_{self.config.experiment.number}", "val_stats_log.txt")
        log_data(log_msg, log_filename)


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

    stats = {
        "mode": mode,
        "episode": episode,
        "kl_loss": kl_loss,
        "total_val_loss": total_val_loss
        }

    return leo, optimizer, stats

