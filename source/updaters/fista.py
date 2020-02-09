from utils import util
import torch.nn as nn
import sys
sys.path.append('../')


class Fista(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Fista, self).__init__()
        self.phi = phi
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        last_x, y, phi, last_t = params
        frame = phi.shape[0]
        y_t = last_x.mul(phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(frame, 1, 1,
                           1).permute(1, 0, 2, 3).mul(phi)
        t = (1+(1+4*last_t**2)**0.5)/2
        x = x + (last_t-1)/t*(x-last_x)
        x = self.denoiser(x)
        return x, y, phi, t

    def initialize(self,phi):
        t = 1
        return [t]
