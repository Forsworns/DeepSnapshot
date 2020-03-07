import torch
import torch.nn as nn
from torch.nn import MSELoss

class LayerLoss(nn.Module):
    def __init__(self, t_layer):
        super(LayerLoss, self).__init__()
        self.t = t_layer

    def forward(self, layers, truth):
        loss = MSELoss(layers[-1],truth)
        for l in layers:
            loss += self.t * MSELoss(l,truth)        
        return loss
