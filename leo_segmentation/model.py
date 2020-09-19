# Architecture definition
# Computational graph creation
import torch
from torch import nn
import numpy as np
from torch.distributions import Normal
from utils import log_data, one_hot_target, calc_iou_per_class, display_data_shape, get_named_dict, tensor_to_numpy, \
    list_to_tensor
import os
from tqdm import tqdm


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        ]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))

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

    def __init__(self, config, mode="meta_train"):
        super(LEO, self).__init__()
        self.config = config
        self.mode = mode
        self.loss = nn.CrossEntropyLoss()
        self.enc1 = _EncoderBlock(14, 16, dropout=True)
        self.enc2 = _EncoderBlock(16, 28, dropout=True)
        self.enc3 = _EncoderBlock(28, 32, dropout=True)
        self.dec3 = _DecoderBlock(32, 28)
        self.dec2 = _DecoderBlock(28, 16)
        self.dec1 = _DecoderBlock(16, 14)  # num of channels in decoder output must be equal to input
        self.RelationNetwork = RelationNetwork(64, 32)
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

    def seg_network(self, inputs, kernels):
        return F.conv2d(inputs, kernels, stride=1, padding=1)

    def leo_inner_loop(self, data, latents):
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
        inner_lr = self.config["hyperparameters"]["inner_loop_lr"]
        initial_latents = latents
        tr_loss, _, _ = self.forward_decoder(data, latents)

        for _ in range(self.config["hyperparameters"]["num_adaptation_steps"]):
            tr_loss.backward()
            grad = latents.grad

            with torch.no_grad():
                latents -= inner_lr * grad
            tr_loss.zero_()
            tr_loss, segmentation_weights, iou = self.forward_decoder(data, latents)
        return tr_loss, segmentation_weights, iou

    def finetuning_inner_loop(self, data, leo_loss, segmentation_weights, inference=False):
        """
        This function does "segmentation_weights optimization"
        Args:
            data (dict) : contains tr_data, tr_data_masks, val_data, val_data_masks
            leo_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> tr_data_mask, prediction--> einsum(tr_data, segmentation_weights))
           segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        Returns:
            val_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> val_data_mask, prediction--> einsum(val_data, segmentation_weights))
        """
        finetuning_lr = self.config["hyperparameters"]["finetuning_lr"]
        for _ in range(self.config["hyperparameters"]["num_finetuning_steps"]):
            grad = torch.autograd.grad(leo_loss, segmentation_weights)

            with torch.no_grad():
                segmentation_weights -= finetuning_lr * grad[0]
            leo_loss, _ = self.calculate_inner_loss(data["tr_data"], data["tr_data_masks"], segmentation_weights)
        val_loss_list = []
        if self.mode == "meta_train":
            for i in range(segmentation_weights.shape[0]):
                duplicate = segmentation_weights[i].repeat(self.config["data_params"]["n_val_per_class"][self.mode], 1,
                                                           1, 1)
            val_loss, iou_val = self.calculate_inner_loss(data["val_data"], data["val_data_masks"], duplicate)
        else:
            if inference:
                return _, _, segmentation_weights
            mean_ious = []
            val_losses = []
            val_img_paths = data.val_data
            val_mask_paths = data.val_data_masks
            for _img_path, _mask_path in tqdm(zip(val_img_paths, val_mask_paths)):
                input_embedding = list_to_tensor(_img_path)
                input_mask = list_to_tensor(_mask_path)
                val_loss, prediction = self.calculate_inner_loss(input_embedding, input_mask, segmentation_weights)
                mean_iou = calc_iou_per_class(prediction, input_mask)
                mean_ious.append(mean_iou)
                val_losses.append(tensor_to_numpy(val_loss))
            mean_iou = np.mean(mean_ious)
            val_loss = np.mean(val_losses)
        # val_loss_list.append(val_loss.item())
        # print("val_loss_list", val_loss_list)
        return val_loss, iou_val, segmentation_weights

    def forward_encoder(self, data):
        """
        This function generates latent codes  from the input data pass it through relation network and
        computes kl_loss after sampling
        Args:
            data (tensor): input or tr_data shape ((num_classes * num_eg_per_class), channel, H, W)
            mode (str): overwrites the default mode "meta_train" with one among ("meta_train", "meta_val", "meta_test")
        Returns:
            latent_leaf (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W) leaf node where
            backpropagation of LEO ends.
            kl_loss (tensor_0shape): how much the distribution sampled from latent code deviates from normal distribution
        """
        self.mode = self.mode
        data = data.to(self.device)
        encoder_output = self.encoder(data)
        # print("encoder_output", encoder_output.shape)
        # print("I am entering relation network")
        relation_network_outputs = self.relation_network(encoder_output)
        # print("I am outside relation network")
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        # print("latent_dist_params", latent_dist_params.shape)
        # latents, kl_loss = self.sample_latents(encoder_output, device)
        latents, kl_loss = self.possibly_sample(latent_dist_params)
        latent_leaf = latents.clone().detach()  # to make latents the leaf node to perform backpropagation until latents
        latent_leaf.requires_grad_(True)
        return latent_leaf, kl_loss

    def forward_decoder(self, data, latents):
        """
        This function decodes the latent codes to get the segmentation weights that has same shape as input tensor
        and computes the leo cross entropy segmentation loss.
        Args:
            data (dict) : contains tr_data, tr_data_masks, val_data, val_data_masks
            latents (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W)
        Returns:
            loss (tensor_0shape): computed as crossentropyloss (groundtruth--> tr/val_data_mask, prediction--> einsum(tr/val_data, segmentation_weights))
            segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        """
        # removed kl divergence sampling from decoder
        segmentation_weights = self.decoder(latents)
        dim_list = list(segmentation_weights.size())
        """
        segmentation_weights= segmentation_weights.permute(1, 2, 3, 0)
        print("segmentation_weights1", segmentation_weights.shape)
        segmentation_weights = segmentation_weights.view(dim_list[1], dim_list[2], dim_list[3], self.config.data_params.num_classes, -1 )
        segmentation_weights = segmentation_weights.permute(3, 4, 0, 1, 2)
        print("segmentation_weights3", segmentation_weights.shape)
        """
        loss, iou = self.calculate_inner_loss(data["tr_data"], data["tr_data_masks"], segmentation_weights)
        return loss, segmentation_weights, iou

    def calculate_inner_loss(self, inputs, true_outputs, segmentation_weights):
        """
        This function finds the prediction mask and calculates the loss of prediction_mask from ground_truth mask
        Args:
             inputs (tensor): tr/val_data shape ((num_classes * num_eg_per_class), channels, H, W)
             true_outputs (tensor): tr/val_data_mask shape ((num_classes * num_eg_per_class), 1 (binary mask), H, W)
             segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        Returns:
            loss (tensor_0shape): computed as crossentropyloss (groundtruth--> tr/val_data_mask, prediction--> einsum(tr/val_data, segmentation_weights))
        """
        seg_net = torch.nn.Sequential(nn.Conv2d(in_channels=14, out_channels=2, kernel_size=3,
                                                stride=1, padding=1)).to(self.device)
        # inputs = inputs.to(self.device)
        if type(true_outputs) == list:
            print("I am here")
        true_outputs = torch.FloatTensor(true_outputs)
        true_outputs2 = true_outputs.squeeze(1).detach().cpu()
        output_mask = seg_net(segmentation_weights.to(self.device))  # self.predict(inputs, segmentation_weights)
        hot_true_outputs = one_hot_target(true_outputs2).to(self.device)
        """
        # output_mask = torch.argmax(prediction, dim=2) #needed to compute accuracy
        dim_list = list(output_mask.s size())
        output_mask = output_mask.permute(2, 3, 4, 0, 1)
        output_mask = output_mask.view(dim_list[2], dim_list[3], dim_list[4], -1)
        output_mask = output_mask.permute(3, 0, 1, 2)
        output_mask = output_mask.type(torch.FloatTensor)
        target = true_outputs.clone()
        true_outputs[target > self.config["data_params"]["num_classes"]-1] = 0 #temporary sample data improper
        """
        loss = self.loss(output_mask.type(torch.FloatTensor), true_outputs.squeeze(1).type(torch.LongTensor))
        iou = calc_iou_per_class(output_mask, true_outputs)
        return loss, iou

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
        # print("concat_codes_serial", concat_codes_serial.shape)
        outputs = self.RelationNetwork.forward(concat_codes_serial)
        # print("I am out out of RelationNetwork.forward")
        dim_list2 = list(outputs.size())
        # print(1)
        outputs = outputs.view(int(np.sqrt(dim_list2[0])), dim_list2[1], dim_list2[2], dim_list2[3],
                               int(np.sqrt(dim_list2[0])))  # torch.cat((output1, output2), dim = 1)
        # print(2)
        outputs = outputs.permute(0, 4, 1, 2, 3)
        # print(3)
        outputs = torch.mean(outputs, dim=1)
        # print("I will return the outputs")
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

    def possibly_sample(self, distribution_params, stddev_offset=0.):
        dim_list = list(distribution_params.size())
        means, stddev = torch.split(distribution_params, int(dim_list[1] / 2), dim=1)
        # stddev = torch.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset)  # try your ideas
        temp = torch.tensor(1e-10)
        temp = temp.to(self.device)
        stddev = torch.max(stddev, temp)
        distribution = Normal(means, stddev)
        if not self.mode == 'meta_train':
            return means, 0.0
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence

    def sample_latents(self, encoder_output, stddev_offset=0.):
        split_size = int(encoder_output.shape[1] // 2)
        splits = [split_size, split_size]
        means, stddev = torch.split(encoder_output, splits, dim=1)
        stddev -= (1. - stddev_offset)  # try your ideas
        temp = torch.tensor(1e-10)
        temp = temp.to(self.device)
        stddev = torch.max(stddev, temp)
        distribution = Normal(means, stddev)
        if not self.mode == 'meta_train':
            return means, 0.0
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence
        # return latents, mean, logvar

    def kl_divergence(self, samples, normal_distribution):
        random_prior = Normal(torch.zeros_like(samples), torch.ones_like(samples))
        kl = torch.mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
        return kl

    def predict(self, inputs, weights):
        """
        inputs (tensor): tr/val_data shape ((num_classes * num_eg_per_class), channels, H, W)
        segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        """
        """
          #inputs = inputs.unsqueeze(2) #num of channels is missing in sample data
          inputs = inputs.permute(1, 0, 4, 2, 3)
          #after_dropout = torch.nn.dropout(inputs, rate=self.dropout_rate)
          # This is 3-dimensional equivalent of a matrix product, where we sum over
          # the last (embedding_dim) dimension. We get [N, K, N, K, H, W] tensor as output.
          per_image_predictions = torch.einsum("ijkab,lmkab->ijlmab", inputs, weights)

          # Predictions have shape [N, K, N]: for each image ([N, K] of them), what
          # is the probability of a given class (N)?
          predictions = torch.mean(per_image_predictions, dim=3)
        """
        predictions = seg_net(weights)
        return predictions

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
        metatrain_gradients = torch.autograd.grad(metatrain_loss, metatrain_variables, retain_graph=True,
                                                  allow_unused=True)
        # nan_loss_or_grad = torch.isnan(metatrain_loss)| torch.reduce([torch.reduce(torch.isnan(g))for g in metatrain_gradients])
        return metatrain_gradients, metatrain_variables

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
            tr_loss, adapted_segmentation_weights, iou_train = self.leo_inner_loop(data_dict, latents)
            # val_loss, mean_iou, seg_weights = self.finetuning_inner_loop(data_dict, tr_loss, adapted_seg_weights, mode)
            val_loss, mean_iou, seg_weights = self.finetuning_inner_loop(data_dict, tr_loss,
                                                                         adapted_segmentation_weights)
            mean_iou_dict[classes[batch]] = mean_iou
            total_val_loss.append(val_loss)
            kl_loss = kl_loss * self.config.hyperparameters.kl_weight
            kl_losses.append(kl_loss)

        total_val_loss = sum(total_val_loss) / len(total_val_loss)
        total_kl_loss = sum(kl_losses) / len(kl_losses)
        total_loss = total_val_loss + total_kl_loss
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
    model_dir = os.path.join(config.data_path, "models", "experiment_{}" \
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
