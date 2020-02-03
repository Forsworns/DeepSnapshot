import torch.nn as nn
import torch
from torch.autograd import Variable

# net is same as dncnn? only with a bias?

# from xc


class SparseNet(nn.Module):
    def __init__(self, channels, **kwgs):
        super(SparseNet, self).__init__()
        features = 64
        kernel_size = 3
        stride = 1
        padding = 1
        if 'pixel' in kwgs:
            pixel = kwgs['pixel']
        else:
            pixel = 256
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        def conv_width(w): return (w+2*padding-kernel_size)/stride+1
        width = int(conv_width(conv_width(conv_width(pixel))))
        limit = Variable(torch.zeros(features, width, width),
                         requires_grad=True)
        self.threshold = lambda x: torch.mul(torch.sign(
            x), nn.functional.relu(torch.abs(x) - limit))

        self.deconv1 = nn.Conv2d(in_channels=features, out_channels=features,
                                 kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv2 = nn.Conv2d(in_channels=features, out_channels=features,
                                 kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.deconv3 = nn.Conv2d(in_channels=features, out_channels=channels,
                                 kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.threshold(x)
        x = self.deconv1(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x
