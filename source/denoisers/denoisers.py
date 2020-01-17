from models.babyunet import BabyUnet
from models.dncnn import DnCNN
from models.singleconv import SingleConvolution
from models.unet import Unet
from models.sparse import SparseNet


def get_denoiser(name, in_channels, out_channels, **kwargs):
    if name == 'unet':
        return Unet(in_channels, out_channels)
    if name == 'baby-unet':
        return BabyUnet(in_channels, out_channels)
    if name == 'dncnn':
        return DnCNN(in_channels, out_channels)
    if name == 'convolution':
        return SingleConvolution(in_channels, out_channels, kwargs['width'])
    if name == 'sparse':
        return SparseNet(in_channels, out_channels)
