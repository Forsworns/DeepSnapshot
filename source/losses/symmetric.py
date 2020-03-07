import torch
import torch.nn as nn


class SymmetricLoss(nn.Module):
    def __init__(self):
        super(SymmetricLoss, self).__init__()

    def forward(self, preds, truth):
        return 0
