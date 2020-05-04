import sys

import torch
import torch.nn as nn

from utils import util

sys.path.append('../')


class Fista(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Fista, self).__init__()
        self.denoiser = denoiser
        self.step_size = step_size
        self.t = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, *params):
        last_x, y, phi, _ = params
        frame = phi.shape[0]
        y_t = last_x.mul(phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(frame, 1, 1,
                           1).permute(1, 0, 2, 3).mul(phi)
        x = x + self.t*(x-last_x)
        x, symmetric = self.denoiser(x)
        return x, y, phi, symmetric

    def initialize(self, phi, cfg):
        return []
