from torch.nn import L1Loss

from losses.layer import LayerLoss
from losses.mse import MSELoss


def get_loss(cfg):
    l_name = cfg.l_name
    if l_name == 'mse':
        return MSELoss()
    elif l_name == 'l1':
        return L1Loss()
    elif l_name == 'layer':
        return LayerLoss(cfg.l_layer)
    elif l_name == 'sparse':
        return SparseLoss()
    elif l_name == 'symetric':
        return SymmetricLoss()
