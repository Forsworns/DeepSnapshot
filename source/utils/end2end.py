import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from denoisers import get_denoiser
from updaters import get_updater

sys.path.append('../')


class End2end(nn.Module):
    def __init__(self, phi, cfg):
        super(End2end, self).__init__()
        channels = phi.shape[0]
        self.layers = nn.ModuleList()
        if cfg.share:
            denoiser = get_denoiser(cfg.d_name, channels)
            for _ in range(cfg.phase):
                step_size = nn.Parameter(0.1*torch.ones(1), requires_grad=True)
                updater = get_updater(cfg.u_name, denoiser, step_size)
                self.layers.append(updater)
        else:
            for _ in range(cfg.phase):
                step_size = nn.Parameter(0.1*torch.ones(1), requires_grad=True)
                denoiser = get_denoiser(cfg.d_name, channels)
                updater = get_updater(cfg.u_name, denoiser, step_size)
                self.layers.append(updater)
        print(self.layers)
        self.initial_params = updater.initialize(phi, cfg)

    def forward(self, x, y, phi):
        layers = []
        symmetric = []
        params = [x, y, phi, None]
        params.extend(self.initial_params)
        for l in self.layers:
            params = l(*params)
            layers.append(params[0])
            symmetric.append(params[3])
        return layers, symmetric
