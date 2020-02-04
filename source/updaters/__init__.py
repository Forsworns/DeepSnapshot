from updaters.ista import Ista
from updaters.fista import Fista
from updaters.gap import Gap
from updaters.admm import Admm
from updaters.amp import Amp


def get_updater(name,phi,denoiser,step_size):
    if name == 'ista':
        return Ista(phi,denoiser,step_size)
    if name == 'fista':
        return Fista(phi,denoiser,step_size)
    if name == 'gap':
        return Gap(phi,denoiser,step_size)
    if name == 'admm':
        return Admm(phi,denoiser,step_size)
    if name == 'amp':
        return Amp(phi,denoiser,step_size)
