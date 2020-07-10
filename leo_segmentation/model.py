# Architecture definition
# Computational graph creation
import torch
import torch.nn as nn

class LEO(nn.Module):
  def __init__(self, lantent_dim):
    super(LEO, self).__init__()
    self.encoder = self.encoder_layers()
    
  def forward(self, x):
      x = self.encoder(x)
      return x

  def encoder_layers(self):
    layers = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), #output(None,32, 14, 14)
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#output(None, 32, 7, 7)
        nn.BatchNorm2d(32),
        nn.ReLU(True),       
        nn.Linear(32*7*7, 2*self.lantent_dim)#dim = 2*lantent_dim
    )
    return layers

   
