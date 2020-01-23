from updaters.fista import Fista
from updaters.gap import Gap
from updaters.admm import Admm
from updaters.amp import Amp


def get_updater(name,phi,y,denoiser,step_size):
    if name == 'fista':
        return Fista(phi,y,denoiser,step_size)
    if name == 'gap':
        return Gap(phi,y,denoiser,step_size)
    if name == 'admm':
        return Admm()
    if name == 'amp':
        return Amp()
