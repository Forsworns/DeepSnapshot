import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channel, conv=default_conv, layer_num=5):
        super(DnCNN, self).__init__()
        kernel_size = 3
        n_feat = 64
        layers = []
        layers.append(conv(channel, n_feat, kernel_size))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(conv(n_feat, n_feat, kernel_size))
            layers.append(nn.BatchNorm2d(n_feat))
            layers.append(nn.ReLU(inplace=True))
        layers.append(conv(n_feat, channel, kernel_size))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
