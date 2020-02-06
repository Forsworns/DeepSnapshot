import torch.nn as nn
import torch


class Plain(nn.Module):
    def __init__(self,phi,denoiser,step_size):
        super(Plain, self).__init__()

    def forward(self,*params):
        x = param
        return x

    def initialize(self):
        return []