import torch.nn as nn
import sys
sys.path.append('../')


class Ista(nn.Module):
    def __init__(self, phi, denoiser, step_size):
        super(Ista, self).__init__()
        self.frame = phi.shape[0]
        self.phi = phi
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        last_x, y = params
        y_t = last_x.mul(self.phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(self.frame, 1, 1,
                           1).permute(1, 0, 2, 3).mul(self.phi)
        x = self.denoiser(x)
        return x, y

    def initialize(self):
        return []
