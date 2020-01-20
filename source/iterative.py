from models.models import get_denoiser
from optimizers.algorithms import get_updater
import torch.nn as nn


class Iterative(nn.Module):
    def __init__(self, steps, step_size, u_name, denoiser, **kwargs):
        super(Iterative, self).__init__()
        self.updater = get_updater(u_name, denoiser, step_size, kwargs)
        self.denoiser = denoiser
        self.steps = steps

    def forward(self, x):
        for sp in xrange(self.steps):
            x = updater(x)
        return x
