import torch.nn as nn
from updaters import get_updater
from denoisers import get_denoiser
import sys
sys.path.append('../')


class End2end(nn.Module):
    def __init__(self, phi, phase, step_size, u_name, d_name, **kwargs):
        super(End2end, self).__init__()
        channels = phi.shape[0]
        self.layers = nn.ModuleList()
        for _ in range(phase):
            denoiser = get_denoiser(d_name, channels)
            updater = get_updater(u_name, phi, y, denoiser, step_size)
            self.layers.append(updater)
 
        self.initial_params = get_updater(u_name, phi, denoiser, step_size).initialize()

    def forward(self, x, y):
        params = [x, y]
        params.extend(self.initial_params)
        for l in self.layers:
            params = l(*params)
        return params[0]
