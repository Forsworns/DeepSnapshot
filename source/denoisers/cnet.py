import torch
import torch.nn as nn
import torch.nn.functional as F

from denoisers.modules import gen_conv, ContextualAttention 

# contextual attention
class CNet(nn.Module):
    def __init__(self, channel):
        super(CNet, self).__init__()
        self.cnum = 32

        self.coarse_generator = CoarseGenerator(channel, self.cnum)
        self.fine_generator = FineGenerator(channel, self.cnum)

    def forward(self, x):
        x_stage1 = self.coarse_generator(x)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1)
        return x_stage1, x_stage2, offset_flow


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(CoarseGenerator, self).__init__()

        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(FineGenerator, self).__init__()

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1):
        # conv branch
        x = self.conv1(x_stage1)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(x_stage1)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow