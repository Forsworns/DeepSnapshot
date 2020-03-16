import torch
import torch.nn as nn

from denoisers.modules import default_conv

# net is same as dncnn? only with a bias?

# from xc


class SNet(nn.Module):
    def __init__(self, channel, conv=default_conv, **kwgs):
        super(SNet, self).__init__()
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
        orig = x
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        coeff = x
        x = self.threshold(x)
        x = self.deconv1(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        # x = ox + x
        symm = self.deconv1(coeff)
        symm = self.relu2(symm)
        symm = self.deconv2(symm) 
        return x, symm
