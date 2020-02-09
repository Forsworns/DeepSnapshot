from denoisers import get_denoiser
from updaters import get_updater
from torch.autograd import Variable
import torch.nn as nn
import torch
import sys
sys.path.append('../')


class End2end(nn.Module):
    def __init__(self, phi, phase, u_name, d_name, b_share):
        super(End2end, self).__init__()
        channels = phi.shape[0]
        self.layers = nn.ModuleList()
        if b_share:
            denoiser = get_denoiser(d_name, channels)
            for _ in range(phase):
                step_size = nn.Parameter(torch.zeros(1),requires_grad=True)
                updater = get_updater(u_name, denoiser,step_size)
                self.layers.append(updater)
        else:
            for _ in range(phase):
                step_size = nn.Parameter(torch.zeros(1),requires_grad=True)
                denoiser = get_denoiser(d_name, channels)
                updater = get_updater(u_name, denoiser,step_size)
                self.layers.append(updater)
        print(self.layers)
        self.initial_params = updater.initialize(phi)

    def forward(self, x, y, phi):
        params = [x, y, phi]
        params.extend(self.initial_params)
        for l in self.layers:
            params = l(*params)
        return params[0]
