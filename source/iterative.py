from denoisers import get_denoiser
from updaters import get_updater
import torch.nn as nn


class Iterative(nn.Module):
    def __init__(self, steps, step_size, u_name, denoiser, **kwargs):
        super(Iterative, self).__init__()
        self.updater = get_updater(u_name, step_size)
        self.denoiser = denoiser
        self.steps = steps

    def forward(self, x):
        if type(updater) is tuple:
            updater1, updater2 = updater
            for sp in xrange(self.steps):
                x = updater1(x)
                x = denoiser(x)
                x = updater2(x)
        else:
            for sp in xrange(self.steps):
                x = updater(x)
                x = denoiser(x)
        return x
