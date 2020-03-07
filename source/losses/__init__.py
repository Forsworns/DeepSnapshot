from torch.nn import L1Loss, MSELoss

from losses.layer import LayerLoss
from losses.sparse import SparseLoss
from losses.symmetric import SymmetricLoss


def get_loss(l_name):
    if l_name == 'mse':
        return MSELoss()
    elif l_name == 'l1':
        return L1Loss()
    elif l_name == 'layer':
        return LayerLoss()
    elif l_name == 'sparse':
        return SparseLoss()
    elif l_name == 'symetric':
        return SymmetricLoss()
