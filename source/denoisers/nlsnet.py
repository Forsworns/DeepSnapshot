import torch.nn as nn

from denoisers.modules import (DownSampler, NLResGroup, ResBlock, ResGroup,
                               UpSampler, default_conv)


# NL scale
class NlsNet(nn.Module):
    def __init__(self, channel, features=16, conv=default_conv):
        super(NlsNet, self).__init__()
        kernel_size = 3
        layers = []
        layers.append(conv(channel, features, kernel_size))
        layers.append(DownSampler(2, features))
        #layers.append(ResGroup(conv, features*4, kernel_size))
        layers.append(NLResGroup(conv, features*4, kernel_size))
        #layers.append(ResGroup(conv, features*4, kernel_size))
        layers.append(conv(features*4, channel, kernel_size))
        layers.append(UpSampler(conv, 2, channel))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        ox = x
        x = self.net(x)
        x = x + ox
        return x
