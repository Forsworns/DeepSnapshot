import torch
import torch.nn as nn


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def forward(self, preds, truth):
        return 0
