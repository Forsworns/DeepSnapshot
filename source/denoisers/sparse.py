import torch.nn as nn
import torch
from torch.autograd import Variable

# net is same as dncnn? only with a bias?

# from xc


class SparseNet(nn.Module):
    def __init__(self, channels):
        super(SparseNet, self).__init__()
        features = 64
        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features,
                               kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=kernel_size, padding=padding, bias=False)
        self.relu1 = nn.Relu(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=kernel_size, padding=padding, bias=False)

        self.deconv1 = nn.Conv2d(in_channels=features, out_channels=features,
                                 kernel_size=kernel_size, padding=padding, bias=False)
        self.relu2 = nn.relu()
        self.deconv2 = nn.Conv2d(in_channels=features, out_channels=features,
                                 kernel_size=kernel_size, padding=padding, bias=False)
        self.deconv3 = nn.Conv2d(in_channels=features, out_channels=channels,
                                 kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        threshold = Variable(x.size())
        self.conv1(x)
        self.conv2(x)
        self.relu1(x)
        self.conv3(x)
        x = torch.mul(torch.sign(x), nn.Relu()(torch.abs(x) - threshold))
        self.deconv1(x)
        self.relu2(x)
        self.deconv2(x)
        self.deconv3(x)
        return x
