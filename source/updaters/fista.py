import torch.nn as nn


class Fista(nn.Module):
    def __init__(self, phi, denoiser, step_size):
        super(Fista, self).__init__()
        self.frame = phi.shape[0]
        self.phi = phi
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        last_x, y, last_t = params
        y_t = last_x.mul(self.phi).sum(1)
        x = last_x - self.step_size * \
            (y_t-y).repeat(self.frame, 1, 1,
                                1).permute(1, 0, 2, 3).mul(self.phi)
        t = (1+(1+4*last_t**2)**0.5)/2
        x = x + (last_t-1)/t*(x-last_x)
        x = self.denoiser(x)
        return x, y, t

    def initialize(self):
        t = 1
        return [t]
