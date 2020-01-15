from models.models import get_updater
from models.models import get_denoiser


def iterative(ites, u_name, d_name, in_channels, out_channels, **kwargs):
    updater = get_updater(u_name)
    denoiser = get_denoiser(d_name, in_channels, out_channels, kwargs)
    for ite in xrange(ites):
        x = updater(x)
        x = denoiser(x)
        
