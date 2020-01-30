import torch.nn as nn
from torch import cat


class SingleConvolution(nn.Module):
    def __init__(self, channel, width=3, torus=False):
        super(SingleConvolution, self).__init__()

        self.torus = torus

        if self.torus:
            self.pad = width // 2
            self.conv = nn.Conv2d(
                channel, channel, kernel_size=width, padding=0)
        else:
            # directly zero pading
            self.conv = nn.Conv2d(
                channel, channel, kernel_size=width, padding=width // 2)

    def forward(self, x):
        if self.torus:
            x = pad_circular(x, self.pad)
            return self.conv(x)
        else:
            return self.conv(x)


def pad_circular(x, pad):
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    if len(x.shape) == 2:
        x = cat([x, x[0:pad]], dim=0)
        x = cat([x, x[:, 0:pad]], dim=1)
        x = cat([x[-2 * pad:-pad], x], dim=0)
        x = cat([x[:, -2 * pad:-pad], x], dim=1)

    elif len(x.shape) == 4:
        x = cat([x, x[:, :, 0:pad]], dim=2)
        x = cat([x, x[:, :, :, 0:pad]], dim=3)
        x = cat([x[:, :, -2 * pad:-pad], x], dim=2)
        x = cat([x[:, :, :, -2 * pad:-pad], x], dim=3)

    return x
