# Architecture definition
# Computational graph creation
import torch
import torch.nn as nn

class LEO(nn.Module):
  def __init__(self):
    super(LEO, self).__init__()
    self.encoder = self.encoder_layers()
    
  def forward(self, x):
      x = self.encoder(x)
      return x

  def encoder_layers(self):
    layers = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1), #output(None,32, 14, 14)     
    )
    return layers

   
