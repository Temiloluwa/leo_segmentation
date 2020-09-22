import torch, os
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from utils import display_data_shape, get_named_dict, one_hot_target, \
    softmax, sparse_crossentropy, calc_iou_per_class, log_data, load_config, \
    summary_write_masks, load_npy, numpy_to_tensor, tensor_to_numpy, list_to_tensor

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        self.dense_input_shape = (28, 192, 256)
        self.latent_size = 50
        super(_EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))

        #layers.extend([Flatten(), nn.Linear(int(np.prod(self.dense_input_shape)), 2 * self.latent_size)])
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        self.dense_input_shape = (28, 192, 256)
        self.latent_size = 50
        super(_DecoderBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))

        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class RelationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(RelationNetwork, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels, momentum=1, affine=True),
                  nn.ReLU(inplace=False),
                  nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(in_channels, momentum=1, affine=True),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        self.relation = nn.Sequential(*layers)

    def forward(self, concat_features):
        out = self.relation(concat_features)
        return out


class LEO(nn.Module):
    """
    contains functions to perform latent embedding optimization
    """

    def __init__(self, config, writer, mode="meta_train"):
        super(LEO, self).__init__()
        self.config = config
        self.mode = mode
        self.writer = writer
        self.enc1 = _EncoderBlock(14, 32, dropout=True)
        self.enc2 = _EncoderBlock(32, 64, dropout=True)
        self.enc3 = _EncoderBlock(64, 128, dropout=True)
        self.RelationNetwork = RelationNetwork(256, 128)
        self.dec3 = _DecoderBlock(128, 64)
        self.dec2 = _DecoderBlock(64, 32)
        self.dec1 = _DecoderBlock(32, 28)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")

    def encoder(self, embeddings):
        enc1 = self.enc1(embeddings)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        return enc3

    def decoder(self, latent):
        dec3 = self.dec3(latent)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        return dec1

    def forward_encoder(self, inputs):
        o = self.encoder(inputs)
        relation_network_outputs = self.relation_network(o)
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        latents, mean, logvar = self.sample_latents(latent_dist_params)
        self.latents, self.mean, self.logvar = latents, mean, logvar
        kl_loss = -0.5 * torch.sum(logvar - mean.pow(2) - logvar.exp() + 1, dim=1)
        kl_loss = torch.mean(kl_loss)
        return latents, kl_loss

    def relation_network(self, latents):
        """
        concatenates each latent with everyother to compute the relation between different input images
        Args:
           latents (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W)
        Returns:
            outputs (tesnor) : shape ((num_classes * num_eg_per_class), channels, H, W)
        Architecture:
        Example:Original input shape (num_classes->3, num_eg_per class->2, Channel->1, H->32, W->30)
                generated latent shape ((num_classes * num_eg_per_class)->6, Channel->64, H->30, W->18)
                left shape ((num_classes * num_eg_per_class)->6, (num_classes * num_eg_per_class)->6, Channel->64, H->30, W->18)
                right shape ((num_classes * num_eg_per_class)->6, (num_classes * num_eg_per_class)->6, Channel->64, H->30, W->18)
                concat_code--> concats left and right along channels shape ((num_classes * num_eg_per_class)->6, (num_classes * num_eg_per_class)->6, Channel->128, H->30, W->18)
                concat_code_serial --> serialise dim 0, 1 to pass it in sequence to the relation network (36, Channel->128, H->30, W->18)
                pass the concat code serial to relatio network get the output tensor ((36, Channel->128, H->30, W->18)
                reshape the tesnor to get dim 0, 1 before serialising ((num_classes * num_eg_per_class)->6, (num_classes * num_eg_per_class)->6, Channel->128, H->30, W->18)
                find the mean along dim 0, 1 to get ((num_classes * num_eg_per_class)->6, Channel->128, H->30, W->18)
        """
        total_num_examples = self.config["data_params"]["num_classes"] * \
                             self.config["data_params"]["n_train_per_class"][self.mode]
        left = latents.clone().unsqueeze(1).repeat(1, total_num_examples, 1, 1, 1)
        right = latents.clone().unsqueeze(0).repeat(total_num_examples, 1, 1, 1, 1)
        concat_codes = torch.cat((left, right), dim=2)
        dim_list1 = list(concat_codes.size())
        concat_codes_serial = concat_codes.permute(2, 3, 4, 0, 1)
        concat_codes_serial = concat_codes_serial.contiguous().view(dim_list1[2], dim_list1[3], dim_list1[4], -1)
        concat_codes_serial = concat_codes_serial.permute(3, 0, 1, 2)
        outputs = self.RelationNetwork.forward(concat_codes_serial)
        dim_list2 = list(outputs.size())
        outputs = outputs.view(int(np.sqrt(dim_list2[0])), dim_list2[1], dim_list2[2], dim_list2[3],
                               int(np.sqrt(dim_list2[0])))  # torch.cat((output1, output2), dim = 1)
        outputs = outputs.permute(0, 4, 1, 2, 3)
        outputs = torch.mean(outputs, dim=1)
        return outputs

    def average_codes_per_class(self, codes):
        """
        Args:
            codes (tensor): output of relation network
        Returns:
            codes averaged along classes
        """
        dim_list1 = list(codes.size())
        codes = codes.permute(1, 2, 3, 0)  # permute to ensure that view is not distructing the tensor structure
        codes = codes.view(dim_list1[1], dim_list1[2], dim_list1[3], self.config["data_params"]["num_classes"],
                           self.config["data_params"]["n_train_per_class"][self.mode])
        codes = codes.permute(3, 4, 0, 1, 2)
        codes = torch.mean(codes, dim=1)
        codes = codes.unsqueeze(1).repeat(1, self.config["data_params"]["n_train_per_class"][self.mode], 1, 1, 1)
        dim_list2 = list(codes.size())
        codes = codes.permute(2, 3, 4, 0, 1)
        codes = codes.contiguous().view(dim_list2[2], dim_list2[3], dim_list2[4], -1)
        codes = codes.permute(3, 0, 1, 2)
        return codes

    def forward_decoder(self, inputs, latents, targets):
        predicted_weights = self.decoder(latents)
        # average codes per class
        predicted_weights = torch.unsqueeze(torch.mean(predicted_weights, 0), 0)
        inner_loss, pred = self.calculate_inner_loss(inputs, targets, predicted_weights)
        return inner_loss, predicted_weights, pred

    def sample_latents(self, encoder_output):
        split_size = int(encoder_output.shape[1] // 2)
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
        pred = torch.cat([channel_zero, channel_one], 1)
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
        tr_loss, _, _ = self.forward_decoder(inputs, latents, target)

        for _ in range(self.config.hyperparameters.num_adaptation_steps):
            latents_grad = torch.autograd.grad(tr_loss, [latents], create_graph=False)[0]
            with torch.no_grad():
                latents -= inner_lr * latents_grad
            tr_loss, segmentation_weights, _ = self.forward_decoder(inputs, latents, target)
        return tr_loss, segmentation_weights

    def finetuning_inner_loop(self, data, tr_loss, seg_weights, mode, inference=False):
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
        for _ in range(self.config.hyperparameters.num_finetuning_steps):
            seg_weights_grad = torch.autograd.grad(tr_loss, [seg_weights])[0]
            with torch.no_grad():
                seg_weights -= finetuning_lr * seg_weights_grad
            tr_loss, _ = self.calculate_inner_loss(data.tr_data, data.tr_data_masks, seg_weights)
        if mode == "meta_train":
            val_loss, predictions = self.calculate_inner_loss(data.val_data, data.val_data_masks, seg_weights)
            mean_iou = calc_iou_per_class(predictions, data.val_data_masks)
        else:
            if inference:
                return _, _, seg_weights
            mean_ious = []
            val_losses = []
            val_img_paths = data.val_data
            val_mask_paths = data.val_data_masks
            for _img_path, _mask_path in tqdm(zip(val_img_paths, val_mask_paths)):
                input_embedding = list_to_tensor(_img_path)
                input_mask = list_to_tensor(_mask_path)
                val_loss, prediction = self.calculate_inner_loss(input_embedding, input_mask, seg_weights)
                mean_iou = calc_iou_per_class(prediction, input_mask)
                mean_ious.append(mean_iou)
                val_losses.append(tensor_to_numpy(val_loss))
            mean_iou = np.mean(mean_ious)
            val_loss = np.mean(val_losses)
        return val_loss, mean_iou, seg_weights

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

            tr_loss, adapted_seg_weights = self.leo_inner_loop( \
                data_dict.tr_data, latents, data_dict.tr_data_masks)

            val_loss, mean_iou, seg_weights = self.finetuning_inner_loop(data_dict, tr_loss, \
                                                                         adapted_seg_weights, mode)
            mean_iou_dict[classes[batch]] = mean_iou
            total_val_loss.append(val_loss)
            kl_loss = kl_loss * self.config.hyperparameters.kl_weight
            kl_losses.append(kl_loss)
        total_val_loss = sum(total_val_loss) / len(total_val_loss)
        total_kl_loss = sum(kl_losses) / len(kl_losses)
        total_loss = total_val_loss + total_kl_loss
        if mode == "meta_train":
            self.writer.add_scalar('meta_train_loss', total_val_loss, train_stats.episode)
            self.writer.add_scalar('kl_loss', total_kl_loss, train_stats.episode)
        if mode == "meta_val":
            self.writer.add_scalar('meta_val_loss', total_val_loss, train_stats.episode)


        stats_data = {
            "mode": mode,
            "kl_loss": total_kl_loss,
            "total_val_loss": total_val_loss,
            "mean_iou_dict": mean_iou_dict
        }
        train_stats.update_stats(**stats_data)
        return total_loss, train_stats, seg_weights


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
        'mean_iou_dict': stats.mean_iou_dict,

    }

    experiment = config.experiment
    model_root = os.path.join(os.path.dirname(__file__), config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))
    if not os.path.exists(os.path.join(model_dir, "config.txt")):
        torch.save(config, os.path.join(model_dir, "config.txt"))
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


def load_model(config, writer, device):
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
    model_dir = os.path.join(os.path.dirname(__file__), config.data_path, "models", "experiment_{}" \
                             .format(experiment.number))

    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    # if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg = f"\n*********** checkpoint {episode} was loaded **************"
    log_data(msg, log_filename)

    leo = LEO(config, writer).to(device)
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
