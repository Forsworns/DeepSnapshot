import torch.nn as nn


class Admm(nn.Module):
    def __init__(self, phi, denoiser, step_size):
        super(Admm, self).__init__()

    def forward(self, *params):
        x, = params
        return x,

    def initialize(self):
        return []
