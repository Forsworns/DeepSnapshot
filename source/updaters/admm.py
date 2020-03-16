import torch.nn as nn


class Admm(nn.Module):
    def __init__(self, denoiser, step_size):
        super(Admm, self).__init__()
        self.denoiser = denoiser

    def forward(self, *params):
        x, y, phi = params
        x, symmetric = self.denoiser(x)
        return x, y, phi, symmetric


    def initialize(self, phi, cfg):
        return []
