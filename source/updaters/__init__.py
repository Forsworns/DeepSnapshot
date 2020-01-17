from updaters.fista import Fista
from updaters.gap import Gap
from updaters.admm import Admm
from updaters.amp import Amp


def get_updater(name):
    if name == 'fista':
        return Fista()
    if name == 'gap':
        return Gap()
    if name == 'admm':
        return Admm()
    if name == 'amp':
        return Amp()
