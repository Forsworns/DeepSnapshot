import torch.nn as nn
from denoisers.modules import default_conv, ResGroup


class ANet(nn.Module):
    def __init__(self, channels, features=64, num_of_layers=3, conv=default_conv):
        super(ANet, self).__init__()
        kernel_size = 3
        layers = []
        # define head module
        layers.append(conv(channels, features, kernel_size))
        # define body module
        for _ in range(num_of_layers - 2):
            layers.append(ResGroup(conv, features, kernel_size))
        layers.append(conv(features, channels, kernel_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x