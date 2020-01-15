from models.models import get_denoiser
from optimizers.algorithms import get_updater
import torch.nn as nn

class Iterative(nn.Module):
    def __init__(self, steps, step_size, u_name, d_name, in_channels, out_channels, **kwargs):
        super(Iterative, self).__init__()

        self.updater = get_updater(u_name, step_size)
        self.denoiser = get_denoiser(d_name, in_channels, out_channels, kwargs)
        self.steps = steps
    
    def forward(self,x):
        if type(updater) is not tuple:
            for ite in xrange(ites):
                x = updater(x)
                x = denoiser(x)
        else:
            pass
        return x
        
        
