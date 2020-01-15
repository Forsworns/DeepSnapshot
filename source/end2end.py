from models.models import get_denoiser
from optimizers.algorithms import get_updater
import torch.nn as nn

class End2end(nn.Module):
    def __init__(self, depth, u_name, d_name, in_channels, out_channels, **kwargs):
        super(End2end, self).__init__()

        layers = []
        for _ in xrange(self.depth):
            updater = get_updater(u_name)
            denoiser = get_model(d_name, in_channels, out_channels, kwargs)
            if type(updater) is not tuple:
                layers.append(updater)
                layers.append(denoiser)
            else:
                pass

        self.end2end = nn.Sequential(*layers)

    def forward(self, x):
        out = self.end2end(x)
        return out