import torch
import torch.nn as nn

from denoisers.modules import GatedConv

# gated conv

class GNet(nn.Module):
    def __init__(self, channel, conv=GatedConv, layer_num=3):
        super(DNet, self).__init__()
        kernel_size = 3
        n_feat = 64
        layers = []
        layers.append(conv(channel, n_feat, kernel_size))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(layer_num - 2):
            layers.append(conv(n_feat, n_feat, kernel_size))
            layers.append(nn.BatchNorm2d(n_feat))
            layers.append(nn.ReLU(inplace=True))
        layers.append(conv(n_feat, channel, kernel_size))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out