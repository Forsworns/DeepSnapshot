from denoisers import get_denoiser
from updaters import get_updater
import torch.nn as nn


class End2end(nn.Module):
    def __init__(self, phase, u_name, d_name, channels, **kwargs):
        super(End2end, self).__init__()

        layers = []
        for _ in range(self.phase):
            denoiser = get_model(d_name, channels, kwargs)
            updater = get_updater(u_name, denoiser, step_size, kwargs)
            layers.append(updater)

        self.end2end = nn.Sequential(*layers)

    def forward(self, params):
        params = self.end2end(params)
        return params

