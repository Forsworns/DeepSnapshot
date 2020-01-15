import torch.nn as nn

class Fista(nn.Module):
    def __init__(self):
        super(Fista,self).__init__()

    def forward(self, x):
        return x    