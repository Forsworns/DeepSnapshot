import torch.nn as nn
import sys
sys.path.append('../')


class Ista(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Ista, self).__init__()
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        last_x, y, phi = params
        frame = phi.size()[0]
        y_t = last_x.mul(phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(frame, 1, 1,
                           1).permute(1, 0, 2, 3).mul(phi)
        x = self.denoiser(x)
        return x, y, phi

    def initialize(self,phi):
        return []
