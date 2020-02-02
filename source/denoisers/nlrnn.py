import torch.nn as nn
from denoisers.modules import NRResBlock


class NLRNN(nn.Module):
    def __init__(self,  channels, state_num=5):
        super(NLRNN, self).__init__()
        self.state_num = state_num

        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(
                channels, 128, kernel_size=3, padding=1)
        self.nr_res = NRResBlock(128)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
                128, channels, kernel_size=3, padding=1)

    def forward(self, x):
        self.norm1(x)
        self.conv1(x)
        x, corr = self.nr_res(x)
        for i in range(self.state_num-1):
            x, corr = self.nr_res(x, corr) # share the same weights
        self.norm2(x)
        self.activate(x)
        self.conv2(x)
        return x

