import torch
import torch.nn as nn
from torch.nn import MSELoss

class SymmetricLoss(nn.Module):
    def __init__(self, t_symmetric):
        super(SymmetricLoss, self).__init__()
        self.mse = MSELoss()
        self.t = t_symmetric
        
    def forward(self, layers, truth):
        loss = self.mse(layers[-1],truth)
        return loss
