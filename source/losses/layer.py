import torch
import torch.nn as nn
from torch.nn import MSELoss

class LayerLoss(nn.Module):
    def __init__(self):
        super(LayerLoss, self).__init__()

    def forward(self, preds, truth):
        return 0
