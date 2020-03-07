import torch
import torch.nn as nn

from denoisers.modules import GatedConv

# gated conv
class GNet(nn.Module):
    def __init__(self, channel, conv=GatedConv, **kwgs):
        super(GNet, self).__init__()
        n_feat = 64
        kernel_size = 3
        if 'pixel' in kwgs:
            pixel = kwgs['pixel']
        else:
            pixel = 256
        self.conv1 = conv(channel, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = conv(n_feat, n_feat, kernel_size)

        padding = kernel_size//2
        stride = 1

        self.limit = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.threshold = lambda x: torch.mul(torch.sign(
            x), nn.functional.relu(torch.abs(x) - self.limit))

        self.deconv1 = conv(n_feat, n_feat, kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv2 = conv(n_feat, n_feat, kernel_size)
        self.deconv3 = conv(n_feat, channel, kernel_size)

    def forward(self, x):
        ox = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.threshold(x)
        x = self.deconv1(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = ox + x
        return x


'''
class GNet(nn.Module):
    def __init__(self, channel, conv=GatedConv, layer_num=3):
        super(GNet, self).__init__()
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
'''