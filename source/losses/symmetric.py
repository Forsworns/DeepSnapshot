import torch
import torch.nn as nn
from torch.nn import MSELoss

class SymmetricLoss(nn.Module):
    def __init__(self, cfg):
        super(SymmetricLoss, self).__init__()
        self.mse = MSELoss()
        self.t1 = cfg.l_layer
        self.t2 = cfg.l_symmetric
        
    def forward(self, layers, truth, symm):
        loss = self.mse(layers[-1],truth)
        for l in layers:
            loss += self.t1 * self.mse(l,truth)   
        for s in symm:
            loss += self.t2 * self.mse(symm,torch.zeros_like(symm))
        return loss
