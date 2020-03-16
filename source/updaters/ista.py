import sys

import torch.nn as nn

sys.path.append('../')


class Ista(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Ista, self).__init__()
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        last_x, y, phi, _ = params
        frame = phi.size()[0]
        y_t = last_x.mul(phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(frame, 1, 1,
                           1).permute(1, 0, 2, 3).mul(phi)
        x, symmetric = self.denoiser(x)
        return x, y, phi, symmetric

    def initialize(self, phi, cfg):
        return []
