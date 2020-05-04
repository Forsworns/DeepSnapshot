import torch
import torch.nn as nn

from denoisers.modules import GatedConv, ConvBlock

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
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = conv(n_feat, n_feat, kernel_size)

        self.limit = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.threshold = lambda x: torch.mul(torch.sign(
            x), nn.functional.relu(torch.abs(x) - self.limit))

        self.deconv1 = conv(n_feat, n_feat, kernel_size)
        # self.relu2 = nn.ReLU(inplace=True)
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
        x = ox + x
        symm = self.deconv1(coeff)
        symm = self.relu2(symm)
        symm = self.deconv2(symm) 
        return x, symm

'''
class GNet(nn.Module):
    def __init__(self, channel, residual=True, down='conv', up='tconv', activation='selu'):
        super(GNet, self).__init__()

        self.residual = residual

        # down sample
        self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, groups=32)
        self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, groups=64)
        self.down3 = nn.Conv2d(
            128, 128, kernel_size=2, stride=2, groups=128)
        self.down4 = nn.Conv2d(
            256, 256, kernel_size=2, stride=2, groups=256)

        self.down1.weight.data = 0.01 * self.down1.weight.data + 0.25
        self.down2.weight.data = 0.01 * self.down2.weight.data + 0.25
        self.down3.weight.data = 0.01 * self.down3.weight.data + 0.25
        self.down4.weight.data = 0.01 * self.down4.weight.data + 0.25

        self.down1.bias.data = 0.01 * self.down1.bias.data + 0
        self.down2.bias.data = 0.01 * self.down2.bias.data + 0
        self.down3.bias.data = 0.01 * self.down3.bias.data + 0
        self.down4.bias.data = 0.01 * self.down4.bias.data + 0

        self.up1 = nn.ConvTranspose2d(
            256, 256, kernel_size=2, stride=2, groups=256)
        self.up2 = nn.ConvTranspose2d(
            128, 128, kernel_size=2, stride=2, groups=128)
        self.up3 = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2, groups=64)
        self.up4 = nn.ConvTranspose2d(
            32, 32, kernel_size=2, stride=2, groups=32)

        self.up1.weight.data = 0.01 * self.up1.weight.data + 0.25
        self.up2.weight.data = 0.01 * self.up2.weight.data + 0.25
        self.up3.weight.data = 0.01 * self.up3.weight.data + 0.25
        self.up4.weight.data = 0.01 * self.up4.weight.data + 0.25

        self.up1.bias.data = 0.01 * self.up1.bias.data + 0
        self.up2.bias.data = 0.01 * self.up2.bias.data + 0
        self.up3.bias.data = 0.01 * self.up3.bias.data + 0
        self.up4.bias.data = 0.01 * self.up4.bias.data + 0

        self.conv1 = ConvBlock(channel, 32, residual, activation, gated=True)
        self.conv2 = ConvBlock(32, 64, residual, activation, gated=True)
        self.conv3 = ConvBlock(64, 128, residual, activation, gated=True)
        self.conv4 = ConvBlock(128, 256, residual, activation, gated=True)

        self.conv5 = ConvBlock(256, 256, residual, activation, gated=True)

        self.conv6 = ConvBlock(2 * 256, 128, residual, activation, gated=True)
        self.conv7 = ConvBlock(2 * 128, 64, residual, activation, gated=True)
        self.conv8 = ConvBlock(2 * 64, 32, residual, activation, gated=True)
        self.conv9 = ConvBlock(2 * 32, channel, residual, activation, gated=True)

        if self.residual:
            self.convres = ConvBlock(
                channel, channel, residual, activation, gated=True)

    def forward(self, x):
        c0 = x
        c1 = self.conv1(x)
        x = self.down1(c1)
        c2 = self.conv2(x)
        x = self.down2(c2)
        c3 = self.conv3(x)
        x = self.down3(c3)
        c4 = self.conv4(x)
        x = self.down4(c4)
        x = self.conv5(x)
        x = self.up1(x)
        # print("shapes: c0:%sx:%s c4:%s " % (c0.shape,x.shape,c4.shape))
        x = torch.cat([x, c4], 1)  # x[:,0:128]*x[:,128:256],
        x = self.conv6(x)
        x = self.up2(x)
        x = torch.cat([x, c3], 1)  # x[:,0:64]*x[:,64:128],
        x = self.conv7(x)
        x = self.up3(x)
        x = torch.cat([x, c2], 1)  # x[:,0:32]*x[:,32:64],
        x = self.conv8(x)
        x = self.up4(x)
        x = torch.cat([x, c1], 1)  # x[:,0:16]*x[:,16:32],
        x = self.conv9(x)
        if self.residual:
            x = torch.add(x, self.convres(c0))

        return x
        '''