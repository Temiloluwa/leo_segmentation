# Architecture definition
# Computational graph creation
import torch
from torch import nn
from torch.autograd import variable
import numpy as np
from torch.distributions import Normal
import torch.optim as optim
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.decode(x)

class RelationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(RelationNetwork, self).__init__()
        layers = [  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels, momentum=1, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels, momentum=1, affine=True),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout())
        self.relation = nn.Sequential(*layers)


    def forward(self, concat_features):
        out = self.relation(concat_features)
        return out

class LEO(nn.Module):
    def __init__(self, config):
        super(LEO, self).__init__()
        self.config = config
        self.loss = nn.CrossEntropyLoss()
        self.enc1 = _EncoderBlock(1, 32)
        self.enc2 = _EncoderBlock(32, 64, dropout=True)
        self.dec2 = _DecoderBlock(64, 32)
        self.dec1 = _DecoderBlock(32, 1) #num of channels in decoder output must be equal to input * 2


    def encoder(self, embeddings):
        enc1 = self.enc1(embeddings)
        enc2 = self.enc2(enc1)
        return enc2

    def decoder(self, latent):
        dec2 = self.dec2(latent)
        dec1 = self.dec1(dec2)
        return dec1

    def leo_inner_loop(self, data, latents):
        inner_lr = self.config['hyperparameters']['inner_loop_lr']
        initial_latents = latents
        tr_loss, _ = self.forward_decoder(data, latents)
        for _ in range(self.config["hyperparameters"]["num_adaptation_steps"]):
            tr_loss.backward()
            grad = latents.grad
            with torch.no_grad():
                latents -= inner_lr * grad
            tr_loss.zero_()
            tr_loss, segmentation_weights = self.forward_decoder(data, latents)
        return tr_loss, segmentation_weights

    def finetuning_inner_loop(self, data, leo_loss, segmentation_weights):
        finetuning_lr = self.config['hyperparameters']['finetuning_lr']
        for _ in range(self.config["hyperparameters"]["num_finetuning_steps"]):
            segmentation_weights.register_hook(self.save_grad(segmentation_weights))
            leo_loss.backward(retain_graph=True)
            #grad = torch.autograd.grad(segmentation_weights, leo_loss, grad_outputs=segmentation_weights)
            grad = segmentation_weights.grad
            with torch.no_grad():
                segmentation_weights -= finetuning_lr * grad
            leo_loss = self.calculate_inner_loss(data["tr_data_orig"], data["tr_data_masks"], segmentation_weights)
            val_loss = self.calculate_inner_loss(data['val_data_orig'], data["val_data_masks"], segmentation_weights)

        return val_loss

    def forward_encoder(self, data):

        latent = self.encoder(data)
        relation_network_outputs = self.relation_network(latent)
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        latents, kl = self.possibly_sample(latent_dist_params)
        latents.requires_grad_(True)
        return latents, kl

    def forward_decoder(self, data, latents):
        #removed kl divergence sampling from decoder
        segmentation_weights = self.decoder(latents)
        dim_list = list(segmentation_weights.size())
        segmentation_weights= segmentation_weights.permute(1, 2, 3, 0)
        segmentation_weights = segmentation_weights.view(dim_list[1], dim_list[2], dim_list[3], self.config["data_type"]["train"]["num_classes"], -1 )
        segmentation_weights = segmentation_weights.permute(3, 4, 0, 1, 2)
        loss = self.calculate_inner_loss(data["tr_data_orig"], data["tr_data_masks"], segmentation_weights)
        return loss, segmentation_weights

    def calculate_inner_loss(self, inputs, true_outputs, segmentation_weights):
        output_mask = self.predict(inputs, segmentation_weights)
        # output_mask = torch.argmax(prediction, dim=2) #needed to compute accuracy
        dim_list = list(output_mask.size())
        output_mask = output_mask.permute(2, 3, 4, 0, 1)
        output_mask = output_mask.view(dim_list[2], dim_list[3], dim_list[4], -1)
        output_mask = output_mask.permute(3, 0, 1, 2)
        output_mask = output_mask.type(torch.FloatTensor)
        target = true_outputs.clone()
        true_outputs[target > 7] = 0 #temporary sample data improper
        loss = self.loss(output_mask.type(torch.FloatTensor), true_outputs.squeeze(1).type(torch.LongTensor))
        return loss

    def relation_network(self, latents):
        total_num_examples = self.config["data_type"]["train"]["num_classes"] * self.config["data_type"]["train"]["n_train_per_class"]
        left = latents.unsqueeze(1).repeat(1, total_num_examples, 1, 1, 1)
        right = latents.unsqueeze(0).repeat(total_num_examples, 1, 1, 1, 1)
        concat_codes = torch.cat((left, right), dim = 2)
        dim_list1 = list(concat_codes.size())
        concat_codes_serial = concat_codes.permute(2, 3, 4, 0, 1)
        concat_codes_serial = concat_codes_serial.contiguous().view(dim_list1[2], dim_list1[3], dim_list1[4], -1)
        concat_codes_serial = concat_codes_serial.permute(3, 0, 1, 2)
        model = RelationNetwork(128, 64)
        outputs = RelationNetwork.forward(model, concat_codes_serial)
        dim_list2 = list(outputs.size())
        outputs = outputs.view(int(np.sqrt(dim_list2[0])), dim_list2[1], dim_list2[2], dim_list2[3], int(np.sqrt(dim_list2[0])))  #torch.cat((output1, output2), dim = 1)
        outputs = outputs.permute(0, 4, 1, 2, 3)
        outputs = torch.mean(outputs, dim=1)

        return outputs

    def average_codes_per_class(self, codes):
        dim_list1 = list(codes.size())
        codes = codes.permute(1, 2, 3, 0) #permute to ensure that view is not distructing the tensor structure
        codes = codes.view(dim_list1[1], dim_list1[2], dim_list1[3], self.config["data_type"]["train"]["num_classes"],
                           self.config["data_type"]["train"]["n_train_per_class"])
        codes = codes.permute(3, 4, 0, 1, 2)
        codes = torch.mean(codes, dim = 1)
        codes = codes.unsqueeze(1).repeat(1, self.config["data_type"]["train"]["n_train_per_class"], 1, 1, 1)
        dim_list2 = list(codes.size())
        codes = codes.permute(2, 3, 4, 0, 1)
        codes = codes.contiguous().view(dim_list2[2], dim_list2[3], dim_list2[4], -1)
        codes = codes.permute(3, 0, 1, 2)
        return codes

    def possibly_sample(self, distribution_params, stddev_offset=0.):
        dim_list = list(distribution_params.size())
        means, unnormalized_stddev = torch.split(distribution_params, int(dim_list[1]/2), dim = 1)
        stddev = torch.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset) #try your ideas
        stddev = torch.max(stddev, torch.tensor(1e-10))
        distribution = Normal(means, stddev)
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence

    def kl_divergence(self, samples, normal_distribution):
        random_prior = Normal(torch.zeros_like(samples), torch.ones_like(samples))
        kl = torch.mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
        return kl

    def predict(self, inputs, weights):
        inputs = inputs.unsqueeze(2) #num of channels is missing in sample data
        #after_dropout = torch.nn.dropout(inputs, rate=self.dropout_rate)
        # This is 3-dimensional equivalent of a matrix product, where we sum over
        # the last (embedding_dim) dimension. We get [N, K, N, K, H, W] tensor as output.
        per_image_predictions = torch.einsum("ijkab,lmkab->ijlmab", inputs, weights)

        # Predictions have shape [N, K, N]: for each image ([N, K] of them), what
        # is the probability of a given class (N)?
        predictions = torch.mean(per_image_predictions, dim=3)
        return predictions


    def save_grad(self, name):
        def hook(grad):
            name.grad = grad
        return hook