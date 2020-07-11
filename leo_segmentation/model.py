# Architecture definition
# Computational graph creation
import torch
from torch import nn
from torch.autograd import variable
import numpy as np
from torch.distributions import Normal


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decode(x)

class RelationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(RelationNetwork, self).__init__()
        layers = [  nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
                    nn.BatchNorm2d(out_channels, momentum=1, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, in_channels, kernel_size=2, padding=1),
                    nn.BatchNorm2d(in_channels, momentum=1, affine=True),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout())
        self.relation = nn.Sequential(*layers)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, concat_features):
        out = self.relation(concat_features)
        out = self.upsample(out)
        return out

class LEO(nn.Module):
    def __init__(self, config):
        super(LEO, self).__init__()
        self.config = config
        self.enc1 = _EncoderBlock(1, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256, dropout=True)

        self.center = _DecoderBlock(256, 512)
        self.dec4 = _DecoderBlock(512, 256)
        self.dec3 = _DecoderBlock(256, 128)
        self.dec2 = _DecoderBlock(128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def encoder(self, embeddings):
        enc1 = self.enc1(embeddings)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        latent = self.center(enc4)
        return latent

    def decoder(self, latent):
        dec4 = self.dec4(latent)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        return dec1

    def forward_encoder(self, data):

        latent = self.encoder(data)
        latent = latent.clone().detach().requires_grad_(True)
        relation_network_outputs = self.relation_network(latent)
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        latents, kl = self.possibly_sample(latent_dist_params)
        return latents, kl

    def forward_decoder(self, data, latents):
        weights_dist_params = self.decoder(latents)
        dim_list = list(latents.size())
        fan_in = dim_list[1]*dim_list[2]*dim_list[3]
        fan_out = self.config["data_type"]["train"]["num_classes"]
        stddev_offset = np.sqrt(2. / (fan_out + fan_in))
        classifier_weights, _ = self.possibly_sample(weights_dist_params,
                                                     stddev_offset=stddev_offset)
        # add sampler - logic returns classifier weights
        # add inner loss
        return 'tr_loss', 'classifier_weights'

    def relation_network(self, latents):
        total_num_examples = self.config["data_type"]["train"]["num_classes"] * self.config["data_type"]["train"]["n_train_per_class"]
        left = latents.unsqueeze(1).repeat(1, total_num_examples, 1, 1, 1)
        right = latents.unsqueeze(0).repeat(total_num_examples, 1, 1, 1, 1)
        concat_codes = torch.cat((left, right), dim = 2)
        dim_list1 = list(concat_codes.size())
        concat_codes_serial = concat_codes.permute(2, 3, 4, 0, 1)
        concat_codes_serial = concat_codes_serial.contiguous().view(dim_list1[2], dim_list1[3], dim_list1[4], -1)
        concat_codes_serial = concat_codes_serial.permute(3, 0, 1, 2)
        model = RelationNetwork(1024, 512)
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
        stddev -= (1. - stddev_offset)
        stddev = torch.max(stddev, torch.tensor(1e-10))
        distribution = Normal(means, stddev)
        samples = distribution.sample()
        kl_divergence = self.kl_divergence(samples, distribution)
        return samples, kl_divergence

    def kl_divergence(self, samples, normal_distribution):
        random_prior = Normal(torch.zeros_like(samples), torch.ones_like(samples))
        kl = torch.mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
        return kl

    def leo_inner_loop(self, data, latents):
        inner_lr = self.config['hyperparameters']['inner_loop_lr']
        initial_latents = latents

        loss, _ = self.forward_decoder(data, latents)
        for _ in range(self.config["hyperparameters"]["num_adaptation_steps"]):
            loss.backward(retain_graph=True)
            loss_grad = torch.autograd.grad(loss, latents, create_graph=True)
            latents -= inner_lr * loss_grad
            loss, classifier_weights = self.forward_decoder(data, latents)
        return loss, classifier_weights

    def finetuning_inner_loop(self, data, leo_loss, classifier_weights):
        finetuning_lr = self.config['hyperparameters']['finetuning_lr']
        for _ in range(self.config["hyperparameters"]["num_finetuning_steps"]):
            leo_loss.backward(retain_graph=True)
            loss_grad = torch.autograd.grad(leo_loss, classifier_weights, create_graph=True)
            classifier_weights -= finetuning_lr * loss_grad
            #calculate train loss
        #calculate validation loss