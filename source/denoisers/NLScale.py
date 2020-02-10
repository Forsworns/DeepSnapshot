import torch.nn as nn
from denoisers.modules import default_conv, UpSampler, DownSampler, ResBlock, ResGroup, NLResGroup


class NLScale(nn.Module):
    def __init__(self, channel, conv=default_conv):
        super(NLScale, self).__init__()
        kernel_size = 3
        n_feat = 16
        layers = []
        layers.append(conv(channel, n_feat, kernel_size))
        layers.append(DownSampler(2, n_feat))
        layers.append(ResGroup(conv, n_feat*4, kernel_size))
        layers.append(NLResGroup(conv, n_feat*4, kernel_size))
        layers.append(ResGroup(conv, n_feat*4, kernel_size))
        layers.append(conv(n_feat*4, channel, kernel_size))
        layers.append(UpSampler(conv, 2, channel))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        ox = x
        x = self.net(x)
        x = x + ox
        return x
