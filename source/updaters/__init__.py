from updaters.plain import Plain
from updaters.ista import Ista
from updaters.fista import Fista
from updaters.gap import Gap
from updaters.admm import Admm
from updaters.amp import Amp


def get_updater(name,denoiser,step_size):
    if name == 'plain':
        return Plain(denoiser,step_size)
    if name == 'ista':
        return Ista(denoiser,step_size)
    if name == 'fista':
        return Fista(denoiser,step_size)
    if name == 'gap':
        return Gap(denoiser,step_size)
    if name == 'admm':
        return Admm(denoiser,step_size)
    if name == 'amp':
        return Amp(denoiser,step_size)
