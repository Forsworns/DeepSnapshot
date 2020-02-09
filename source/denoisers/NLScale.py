import torch.nn as nn
from denoisers.modules import default_conv, UpSampler, DownSampler


class NLScale(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(NLScale, self).__init__()
        modules = []
        modules.append(DownSampler(2))
        modules.append(UpSampler(2))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        ox = x
        x = self.net(x)
        x = x + ox
        return x
