import torch
import torch.nn as nn
from torch.nn import MSELoss

class LayerLoss(nn.Module):
    def __init__(self, cfg):
        super(LayerLoss, self).__init__()
        self.t = cfg.l_layer
        self.mse = MSELoss()

    def forward(self, layers, truth):
        loss = self.mse(layers[-1],truth)
        for l in layers:
            loss += self.t * self.mse(l,truth)        
        return loss
