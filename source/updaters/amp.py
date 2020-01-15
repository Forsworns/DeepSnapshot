import torch.nn as nn

class Amp(nn.Module):
    def __init__(self):
        super(Amp,self).__init__()

    def forward(self, x):
        return x    