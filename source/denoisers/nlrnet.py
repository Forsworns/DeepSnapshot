import torch.nn as nn
from denoisers.modules import default_conv, UpSampler, DownSampler, NRResBlock

# NLRNN
class NlrNet(nn.Module):
    def __init__(self, channel, conv=default_conv, layer_num=1):
        super(NlrNet, self).__init__()
        kernel_size = 3
        n_feat = 64
        self.layer_num = layer_num
        self.norm1 = nn.BatchNorm2d(channel)
        self.conv1 = conv(channel, n_feat, kernel_size)
        self.nr_res = NRResBlock(n_feat, channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = conv(channel, n_feat, kernel_size)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        x, corr = self.nr_res(x)
        for i in range(self.layer_num-1):
            x, corr = self.nr_res(x, corr)  # share the same weights
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x
