import torch.nn as nn


class Amp(nn.Module):
    def __init__(self, phi, denoiser, step_size):
        super(Amp, self).__init__()

    def forward(self, *params):
        x, = params
        return x,

    def initialize(self):
        return []