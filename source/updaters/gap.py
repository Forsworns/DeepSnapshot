import torch.nn as nn

class Gap(nn.Module):
    def __init__(self):
        super(Gap,self).__init__()

    def forward(self, x):
        return x    