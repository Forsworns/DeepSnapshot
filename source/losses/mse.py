import torch
import torch.nn as nn
from torch.nn import MSELoss

class MSELoss(nn.Module):
    def __init__(self, t_layer):
        super(LayerLoss, self).__init__()
        self.mse = MSELoss()
        
    def forward(self, layers, truth):
        loss = self.mse(layers[-1],truth)
        return loss
