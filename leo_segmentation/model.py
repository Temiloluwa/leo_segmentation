# Architecture definition
# Computational graph creation
import torch
from torch import nn
from torch.autograd import variable

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
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
        latent = variable(self.encoder(data), requires_grad=True)
        # add relation network logic
        # add sampler
        return latent

    def forward_decoder(self, data, latents):
        weights_dist_params = self.decoder(latents)
        # add sampler - logic returns classifier weights
        # add inner loss
        return 'tr_loss', 'classifier_weights'

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