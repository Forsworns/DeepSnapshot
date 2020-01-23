from updaters import get_updater
import torch.nn as nn


class Iterative(nn.Module):
    def __init__(self, steps, step_size, u_name, denoiser, **kwargs):
        super(Iterative, self).__init__()
        self.updater = get_updater(u_name, denoiser, step_size, kwargs)
        self.denoiser = denoiser
        self.steps = steps

    def forward(self, params):
        for sp in range(self.steps):
            params = updater(params)
        return params
