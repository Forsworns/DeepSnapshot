import torch.nn as nn


class Amp(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Amp, self).__init__()

    def forward(self, *params):
        x, = params
        return x,

    def initialize(self):
        return []