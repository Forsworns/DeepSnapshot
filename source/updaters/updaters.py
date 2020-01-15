from optimizers.fista import Fista
from optimizers.gap import Gap 
from optimizers.admm import Admm
from optimizers.amp import Amp 

def get_updater(name):
    if name == 'fista':
        return Fista()
    if name == 'gap':
        return Gap()
    if name == 'admm':
        return Admm()
    if name == 'amp':
        return Amp()