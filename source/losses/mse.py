import torch
import torch.nn as nn
from torch.nn import MSELoss

class MSELoss(nn.Module):
    def __init__(self, t_layer):
        super(LayerLoss, self).__init__()
        
    def forward(self, layers, truth):
        loss = MSELoss(layers[-1],truth)
        return loss
