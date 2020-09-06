# Architecture definition
# Computational graph creation
import torch
from torch import nn
import numpy as np
from torch.distributions import Normal

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2),
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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.decode(x)

class RelationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(RelationNetwork, self).__init__()
        layers = [  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
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
        self.enc1 = _EncoderBlock(14, 16)
        self.enc2 = _EncoderBlock(16, 32, dropout=True)
        self.dec2 = _DecoderBlock(32, 16)
        self.dec1 = _DecoderBlock(16, 14) #num of channels in decoder output must be equal to input
        self.RelationNetwork = RelationNetwork(64, 32)

    def encoder(self, embeddings):
        enc1 = self.enc1(embeddings)
        enc2 = self.enc2(enc1)
        return enc2

    def decoder(self, latent):
        dec2 = self.dec2(latent)
        dec1 = self.dec1(dec2)
        return dec1

    def leo_inner_loop(self, data, latents, device):
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
        tr_loss, _ = self.forward_decoder(data, latents, device)

        for _ in range(self.config["hyperparameters"]["num_adaptation_steps"]):
            tr_loss.backward()
            grad = latents.grad

            with torch.no_grad():
                latents -= inner_lr * grad
            tr_loss.zero_()
            tr_loss, segmentation_weights = self.forward_decoder(data, latents, device)
        return tr_loss, segmentation_weights

    def finetuning_inner_loop(self, data, leo_loss, segmentation_weights, device):
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
            leo_loss = self.calculate_inner_loss(data["tr_data_orig"], data["tr_data_masks"], segmentation_weights, device)
            val_loss = self.calculate_inner_loss(data["val_data_orig"], data["val_data_masks"], segmentation_weights, device)

        return val_loss

    def forward_encoder(self, data, mode, device):
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
        self.mode = mode
        data = data.to(device)
        latent = self.encoder(data)
        relation_network_outputs = self.relation_network(latent)
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        latents, kl_loss = self.possibly_sample(latent_dist_params, device)
        latent_leaf = latents.clone().detach() #to make latents the leaf node to perform backpropagation until latents
        latent_leaf.requires_grad_(True)
        return latent_leaf, kl_loss

    def forward_decoder(self, data, latents, device):
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
        #removed kl divergence sampling from decoder
        segmentation_weights = self.decoder(latents)
        dim_list = list(segmentation_weights.size())
        segmentation_weights= segmentation_weights.permute(1, 2, 3, 0)
        segmentation_weights = segmentation_weights.view(dim_list[1], dim_list[2], dim_list[3], self.config.data_params.num_classes, -1 )
        segmentation_weights = segmentation_weights.permute(3, 4, 0, 1, 2)
        loss = self.calculate_inner_loss(data["tr_data_orig"], data["tr_data_masks"], segmentation_weights, device)
        return loss, segmentation_weights

    def calculate_inner_loss(self, inputs, true_outputs, segmentation_weights, device):
        """
        This function finds the prediction mask and calculates the loss of prediction_mask from ground_truth mask
        Args:
             inputs (tensor): tr/val_data shape ((num_classes * num_eg_per_class), channels, H, W)
             true_outputs (tensor): tr/val_data_mask shape ((num_classes * num_eg_per_class), 1 (binary mask), H, W)
             segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        Returns:
            loss (tensor_0shape): computed as crossentropyloss (groundtruth--> tr/val_data_mask, prediction--> einsum(tr/val_data, segmentation_weights))
        """
        inputs = inputs.to(device)
        true_outputs = true_outputs.to(device)
        output_mask = self.predict(inputs, segmentation_weights)
        # output_mask = torch.argmax(prediction, dim=2) #needed to compute accuracy
        dim_list = list(output_mask.size())
        output_mask = output_mask.permute(2, 3, 4, 0, 1)
        output_mask = output_mask.view(dim_list[2], dim_list[3], dim_list[4], -1)
        output_mask = output_mask.permute(3, 0, 1, 2)
        output_mask = output_mask.type(torch.FloatTensor)
        target = true_outputs.clone()
        true_outputs[target > self.config["data_params"]["num_classes"]-1] = 0 #temporary sample data improper
        loss = self.loss(output_mask.type(torch.FloatTensor), true_outputs.squeeze(1).type(torch.LongTensor))
    
        return loss

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
        total_num_examples = self.config["data_params"]["num_classes"] * self.config["data_params"]["n_train_per_class"][self.mode]
        left = latents.clone().unsqueeze(1).repeat(1, total_num_examples, 1, 1, 1)
        right = latents.clone().unsqueeze(0).repeat(total_num_examples, 1, 1, 1, 1)
        concat_codes = torch.cat((left, right), dim = 2)
        dim_list1 = list(concat_codes.size())
        concat_codes_serial = concat_codes.permute(2, 3, 4, 0, 1)
        concat_codes_serial = concat_codes_serial.contiguous().view(dim_list1[2], dim_list1[3], dim_list1[4], -1)
        concat_codes_serial = concat_codes_serial.permute(3, 0, 1, 2)
        outputs = self.RelationNetwork.forward(concat_codes_serial)
        dim_list2 = list(outputs.size())
        outputs = outputs.view(int(np.sqrt(dim_list2[0])), dim_list2[1], dim_list2[2], dim_list2[3], int(np.sqrt(dim_list2[0])))  #torch.cat((output1, output2), dim = 1)
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
        codes = codes.permute(1, 2, 3, 0) #permute to ensure that view is not distructing the tensor structure
        codes = codes.view(dim_list1[1], dim_list1[2], dim_list1[3], self.config["data_params"]["num_classes"],
                           self.config["data_params"]["n_train_per_class"][self.mode])
        codes = codes.permute(3, 4, 0, 1, 2)
        codes = torch.mean(codes, dim = 1)
        codes = codes.unsqueeze(1).repeat(1, self.config["data_params"]["n_train_per_class"][self.mode], 1, 1, 1)
        dim_list2 = list(codes.size())
        codes = codes.permute(2, 3, 4, 0, 1)
        codes = codes.contiguous().view(dim_list2[2], dim_list2[3], dim_list2[4], -1)
        codes = codes.permute(3, 0, 1, 2)
        return codes

    def possibly_sample(self, distribution_params, device, stddev_offset=0.):
        dim_list = list(distribution_params.size())
        means, stddev = torch.split(distribution_params, int(dim_list[1]/2), dim = 1)
        #stddev = torch.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset) #try your ideas
        temp = torch.tensor(1e-10)
        temp = temp.to(device)
        stddev = torch.max(stddev, temp)
        distribution = Normal(means, stddev)
        if not self.mode == 'meta_train':
            return means, 0.0
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence

    def kl_divergence(self, samples, normal_distribution):
        random_prior = Normal(torch.zeros_like(samples), torch.ones_like(samples))
        kl = torch.mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
        return kl

    def predict(self, inputs, weights):
        #inputs = inputs.unsqueeze(2) #num of channels is missing in sample data
        inputs = inputs.permute(1, 0, 4, 2, 3)
        #after_dropout = torch.nn.dropout(inputs, rate=self.dropout_rate)
        # This is 3-dimensional equivalent of a matrix product, where we sum over
        # the last (embedding_dim) dimension. We get [N, K, N, K, H, W] tensor as output.
        per_image_predictions = torch.einsum("ijkab,lmkab->ijlmab", inputs, weights)

        # Predictions have shape [N, K, N]: for each image ([N, K] of them), what
        # is the probability of a given class (N)?
        predictions = torch.mean(per_image_predictions, dim=3)
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
        metatrain_gradients = torch.autograd.grad(metatrain_loss, metatrain_variables, retain_graph=True)
        #nan_loss_or_grad = torch.isnan(metatrain_loss)| torch.reduce([torch.reduce(torch.isnan(g))for g in metatrain_gradients])
        return metatrain_gradients, metatrain_variables