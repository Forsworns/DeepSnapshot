import torch.nn as nn
import torch
from torch.autograd import Variable
from denoisers.modules import  default_conv
# net is same as dncnn? only with a bias?

# from xc


class SparseNet(nn.Module):
    def __init__(self, channel, conv=default_conv, **kwgs):
        super(SparseNet, self).__init__()
        n_feat = 64
        kernel_size = 3
        if 'pixel' in kwgs:
            pixel = kwgs['pixel']
        else:
            pixel = 256
        self.conv1 = conv(channels, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = conv(n_feat, n_feat, kernel_size)

        def conv_width(w): return (w+2*padding-kernel_size)/stride+1
        width = int(conv_width(conv_width(conv_width(pixel))))
        limit = Variable(torch.zeros(n_feat, width, width),
                         requires_grad=True)
        self.threshold = lambda x: torch.mul(torch.sign(
            x), nn.functional.relu(torch.abs(x) - limit))

        self.deconv1 = conv(n_feat, n_feat, kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv2 = conv(n_feat, n_feat, kernel_size)
        self.deconv3 = conv(n_feat, channels, kernel_size)

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
