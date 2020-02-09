import torch.nn as nn
from denoisers.modules import default_conv, UpSampler, DownSampler, ResBlock, ResGroup, NLResGroup


class NLScale(nn.Module):
    def __init__(self, channel, conv=default_conv):
        super(NLScale, self).__init__()
        kernel_size=3
        n_feat = 64
        layers = []
        layers.append(conv(channel, n_feat, kernel_size))
        layers.append(DownSampler(2))
        layers.append(NLResGroup(
                conv, n_feat, kernel_size, act=act, res_scale=res_scale))
        layers.append(ResGroup(
                conv, n_feat, kernel_size, act=act, res_scale=res_scale))
        layers.append(NLResGroup(
                conv, n_feat, kernel_size, act=act, res_scale=res_scale))
        layers.append(UpSampler(2))
        layers.append(ResBlock(conv, n_feat, kernel_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        ox = x
        x = self.net(x)
        x = x + ox
        return x
