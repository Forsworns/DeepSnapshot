import torch
import torch.nn as nn

class SymmetricLoss(nn.Module):
    def __init__(self, cfg):
        super(SymmetricLoss, self).__init__()
        self.t1 = cfg.l_layer
        self.t2 = cfg.l_symmetric
        
    def forward(self, layers, truth, symm):
        loss = torch.reduce_mean(torch.sort(layers[-1]-truth))
        for l in layers:
            loss += self.t1 * torch.reduce_mean(torch.sqrt(l-truth))   
        for s in symm:
            loss += self.t2 * torch.reduce_mean(torch.sqrt(s))
        return loss
