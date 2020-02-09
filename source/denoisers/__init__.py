from denoisers.babyunet import BabyUnet
from denoisers.dncnn import DnCNN
from denoisers.unet import Unet
from denoisers.sparse import SparseNet
from denoisers.nlrnn import NLRNN
from denoisers.rnan import RNAN
from denoisers.nlscale import NLScale


def get_denoiser(name, channels, **kwargs):
    if name == 'unet':
        return Unet(channels)
    if name == 'baby-unet':
        return BabyUnet(channels)
    if name == 'dncnn':
        return DnCNN(channels)
    if name == 'sparse':
        return SparseNet(channels)
    if name == 'nlrnn':
        return NLRNN(channels)
    if name == 'rnan':
        return RNAN(channels)
    if name == 'nlscale':
        return NLScale(channels)
