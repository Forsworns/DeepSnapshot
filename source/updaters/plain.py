import torch
import torch.nn as nn


class Plain(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Plain, self).__init__()
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        x, y, phi, _ = params
        x, symmetric = self.denoiser(x)
        return x, y, phi, symmetric

    def initialize(self, phi, cfg):
        return []
