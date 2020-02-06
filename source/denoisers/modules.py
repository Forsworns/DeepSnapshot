import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# RNAN modules
# residual attention + downscale upscale + denoising


class ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(ResGroup, self).__init__()
        modules_body = []
        modules_body.append(ResAttModuleDownUpPlus(
            conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

# nonlocal residual attention + downscale upscale + denoising


class NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(NLResGroup, self).__init__()
        modules_body = []

        modules_body.append(NLResAttModuleDownUpPlus(
            conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # if we don't use group residual, donot remove the following conv
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        #res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# add NonLocalBlock2D
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(
                conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        tx = self.body(x)

        return tx


# define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size,
                                   bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(
            n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx

# define nonlocal mask branch


class NLMaskBranchDownUp(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLMaskBranchDownUp, self).__init__()

        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat, n_feat, 3, stride=2, padding=1))

        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size,
                                   bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(
            n_feat, n_feat, 6, stride=2, padding=2))

        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))

        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=True))

        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)

    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


# define residual attention module
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size,
                                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size,
                                      bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size,
                                    bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


# define nonlocal residual attention module
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True,
                               bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size,
                                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size,
                                        bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size,
                                    bias=True, bn=False, act=nn.ReLU(True), res_scale=1))

        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB = nn.Sequential(*RA_TB)
        self.RA_MB = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


# Unet module
# same convblock (stride=1,kernel_size=3,padding=1)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=True, activation='leakyrelu', transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual
        self.activation = activation
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05)
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        if self.transpose:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout1(x)
        if self.norm1:
            x = self.norm1(x)
        x = self.actfun1(x)
        x = self.conv2(x)
        if self.dropout:
            x = self.dropout2(x)
        if self.norm2:
            x = self.norm2(x)
        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:,
                                                             0:min(ox.shape[1], x.shape[1]), :, :]
        x = self.actfun2(x)
        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))
        return x


# NLRNN Modules
class NRResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NRResBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.non_local = NRNLBlock(in_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.activate2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, in_channels,
                               kernel_size=3, padding=1)

    def forward(self, x, corr=None):
        ox = x
        x = self.norm1(x)
        x = self.activate1(x)
        x, corr_new = self.non_local(x, corr)
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.norm3(x)
        x = self.activate2(x)
        x = self.conv2(x)
        x + ox
        return x, corr_new


class NRNLBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NRNLBlock, self).__init__()
        self.conv_theta = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_phi = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_g = nn.Conv2d(in_channels, in_channels,
                                kernel_size=1, padding=0)
        nn.init.constant_(self.conv_g.weight, 0)

    def forward(self, x, corr):
        ox = x
        x_theta = self.conv_theta(x)
        x_phi = self.conv_phi(x)
        x_g = self.conv_g(x)  # better to be zero initialized
        x_theta = x_theta.reshape(
            x_theta.shape[0], x_theta.shape[1], x_theta.shape[2]*x_theta.shape[3])
        x_phi = x_phi.reshape(
            x_phi.shape[0], x_phi.shape[1], x_phi.shape[2]*x_phi.shape[3])
        x_theta = x_theta.permute(0, 2, 1)
        corr_new = torch.matmul(x_theta, x_phi)
        if corr is not None:
            corr_new += corr
        # normalization for embedded Gaussian
        corr_normed = F.softmax(corr_new)
        x_g = x_g.reshape(x_g.shape[0], x_g.shape[1],
                          x_g.shape[2] * x_g.shape[3])
        x = torch.matmul(x_g, corr_normed)
        x = x.reshape_as(ox)
        x = x+ox
        return x, corr_new
