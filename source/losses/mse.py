import torch
import torch.nn as nn
from torch.nn import MSELoss as MSE

class MSELoss(nn.Module):
    def __init__(self, t):
        super(MSELoss, self).__init__()
        self.mse = MSE()
        
    def forward(self, layers, truth):
        loss = self.mse(layers[-1],truth)
        return loss
