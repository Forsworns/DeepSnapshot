import os
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy import clip, exp
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from torch.optim import SGD, Adam
from torch.utils.data import Dataset

import utils.configs as config


def save(model, psnr, reconstruction, label, args):
    _, _, _, para_dir, recon_dir, model_dir = config.general(args.name)
    save_model(model, model_dir, psnr)
    args_dict = vars(args)
    print(args_dict)
    if "trainer" in args_dict:
        args_dict.pop("trainer")
    if "tester" in args_dict:
        args_dict.pop("tester")
    config_log = config.ConfigLog(args_dict)
    config_log.dump(para_dir)

    recon_dir = "{}/{}".format(recon_dir, int(time()))
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            PSNR = compare_psnr(label[i, j], reconstruction[i, j])
            SSIM = compare_ssim(label[i, j], reconstruction[i, j])
            print("Frame %d, PSNR: %.2f, SSIM: %.2f" %
                  (i*args.frame+j, PSNR, SSIM))
            outImg = np.hstack((label[i, j], reconstruction[i, j]))
            imgRecName = "%s/frame%d_PSNR%.2f.png" % (
                recon_dir, i*args.frame+j, PSNR)
            imgRec = Image.fromarray(
                np.clip(255*outImg, 0, 255).astype(np.uint8))
            imgRec.save(imgRecName)


def save_model(model, directory, psnr):
    torch.save(model.state_dict(), "{}/{}-{}.pkl".format(directory,
                                                         int(time()), np.round(psnr, 2)))


def get_optimizer(o_name, model, lr):
    if o_name == 'adam':
        return Adam(model.parameters(), lr)
    elif o_name == 'sgd':
        return SGD(model.parameters(), lr)


def expand(x, r):
    return np.repeat(np.repeat(x, r, axis=0), r, axis=1)


def show_tensors(tensor, cmap='Greys_r', scale=False, titles=None):
    # cmap: 'Greys_r','magma'
    # tensor: n x t x w x d
    im = tensor.detach().cpu().numpy()
    if scale:
        im = im / im.max()
    else:
        im = clip(im, 0, 1)
    if im.shape[0] == 1:
        show_tensor(plt, im[0], cmap=cmap)
    else:
        amount = im.shape[0]
        fig, ax = plt.subplots(1, amount, sharex='col',
                               sharey='row', figsize=(amount * 4, 4))

        for i in range(amount):
            ax[i].get_xaxis().set_ticks([])
            ax[i].get_yaxis().set_ticks([])
            show_tensor(ax[i], im[i], cmap)
            if titles:
                ax[i].set_title(titles[i])
        fig
    plt.show()


def show_tensor(handle, tensor, cmap):
    # tensor: t x w x d
    if len(tensor.shape) == 2:
        handle.imshow(tensor, cmap=cmap)
    else:
        for t in range(tensor.shape[0]):
            handle.imshow(tensor[t])
            plt.pause(0.1)


def tensor_to_numpy(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        return x

    if x.shape[0] == 1:
        return x[0]
    elif x.shape[0] == 3:
        return x.transpose((1, 2, 0))
    else:
        raise


def show_data(datapt):
    # For datasets of the form (noise1, noise2, ground truth), shows all three concatenated
    show_tensor(torch.cat((datapt[0], datapt[1], datapt[2]), dim=2))


def scale_tensor(x):
    return (x - x.min()) / (x.max() - x.min())


def plot_grid(images, height, width, **kwargs):

    if not isinstance(images, np.ndarray):
        images = np.concatenate([im[np.newaxis] for im in images])
    assert images.shape[0] >= width * height

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Greys_r'

    images = images[:width * height]
    fig, ax = plt.subplots(height, width, sharex='col',
                           sharey='row', figsize=(width * 4, height * 4))
    image_grid = images.reshape(
        height, width, images.shape[1], images.shape[2])

    # axes are in a two-dimensional array, indexed by [row, col]
    for i in range(height):
        for j in range(width):
            if (height > 1):
                ax[i, j].imshow(image_grid[i, j], **kwargs)
                ax[i, j].get_xaxis().set_ticks([])
                ax[i, j].get_yaxis().set_ticks([])
            else:
                ax[j].imshow(image_grid[i, j], **kwargs)
                ax[j].get_xaxis().set_ticks([])
                ax[j].get_yaxis().set_ticks([])
    fig


def show(image, **kwargs):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap=plt.cm.gray, **kwargs)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])


def plot_images(image_list, **kwargs):
    images = np.concatenate([im[np.newaxis] for im in image_list])
    plot_grid(images, 1, len(image_list), **kwargs)


def clamp_tensor(x):
    return torch.clamp(x, 0, 1)


def random_noise(img, params):
    """Parameters for random noise include the mode and the type.

    mode: gaussian, poisson, or gaussian_poisson noise type
    std: std of gaussian
    photons_at_max: at image with intensity 1 has this many photons on average
    clamp: clamp result to [0,1]
    """

    noisy = img

    if params['mode'] == 'poisson' or params['mode'] == 'gaussian_poisson':
        noisy = torch.poisson(
            noisy * params['photons_at_max']) / params['photons_at_max']

    if params['mode'] == 'gaussian' or params['mode'] == 'gaussian_poisson':
        noise = torch.randn(img.size()).to(img.device) * params['std']
        noisy = noise + noisy

    if params['mode'] == 'bernoulli':
        noisy = noisy * torch.bernoulli(torch.ones(noisy.shape) * params['p'])

    if 'clamp' in params and params['clamp']:
        noisy = torch.clamp(noisy, 0, 1)

    return noisy


def test_bernoulli_noise():
    torch.manual_seed(2018)
    p = 0.2
    shape = (10, 1, 100, 100)
    n = 10 * 100 * 100
    img = torch.ones(shape)
    noisy = random_noise(img, {'mode': 'bernoulli', 'p': p})

    var = n * p * (1 - p)

    assert torch.abs(noisy.sum() - p * img.sum()) < 3 * (var ** 0.5)


def psnr(x, x_true, max_intensity=1.0, pad=None, rescale=False):
    '''A function computing the PSNR of a noisy tensor x approximating a tensor x_true.

    It vectorizes over the batch.

    PSNR := 10*log10 (MAX^2/MSE)

    where the MSE is the averaged squared error over all pixels and channels.
    '''

    return 10 * torch.log10((max_intensity ** 2) / mse(x, x_true, pad=pad, rescale=rescale))


def test_psnr():
    std = 0.1
    noise = torch.randn(10, 3, 100, 100) * std
    x_true = torch.ones(10, 3, 100, 100) / 2
    x = x_true + noise
    # MSE should be 0.01. PSNR should be 20.
    assert (torch.abs(psnr(x, x_true) - 20) < 0.1).all()

    x = 256 * x
    x_true = 256 * x_true
    assert (torch.abs(psnr(x, x_true, 256) - 20) < 0.2).all()


def test_mse_rescale():
    y = torch.randn(10, 3, 10, 10)
    x = 10 * y + 7
    assert (mse(x, y, rescale=True) < 1e-5).all()

    # Normalized values are (1, 1, 0, -2) and (1, 1, -1, -1)
    y = torch.Tensor([3, 3, 2, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x = torch.Tensor([5, 5, 0, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    assert mse(x, y, rescale=True).sum() == 0.5


def mse(x, y, pad=None, rescale=False):
    if pad:
        x = x[:, :, pad:-pad, pad:-pad]
        y = y[:, :, pad:-pad, pad:-pad]

    def batchwise_mean(z):
        return z.reshape(z.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1)

    if rescale:
        x = x - batchwise_mean(x)
        y = y - batchwise_mean(y)
        a = batchwise_mean(x * y) / batchwise_mean(x * x)
        x = a * x

    return batchwise_mean((x - y) ** 2).reshape(-1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                              float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def smooth(tensor):
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(tensor.device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor, kernel, stride=1, padding=1)
    return filtered_tensor


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class PercentileNormalizer():
    """Percentile-based image normalization.
    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin=2, pmax=99.8, dtype=np.float32, **kwargs):
        if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
            raise ValueError
        self.pmin = pmin
        self.pmax = pmax
        self.dtype = dtype
        self.kwargs = kwargs

        self.mi = None
        self.ma = None

    def normalize(self, img, channel=1):
        """Percentile-based normalization of raw input image.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        axes = tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img, self.pmin, axis=axes, keepdims=True).astype(
            self.dtype, copy=False)
        self.ma = np.percentile(img, self.pmax, axis=axes, keepdims=True).astype(
            self.dtype, copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def denormalize(self, mean):
        """Undo percentile-based normalization to map restored image to similar range as input image.
        """
        alpha = self.ma - self.mi
        beta = self.mi
        return alpha * mean + beta


def test_percentile_normalizer():
    a = np.arange(1000).reshape(10, 1, 10, 10).astype(np.uint16)

    norm = PercentileNormalizer(pmin=0, pmax=100, dtype=np.float32, clip=False)
    assert norm.normalize(a).min() == 0 and norm.normalize(a).max() == 1

    # gap between 10th and 90th percentile is 100 to 900, so
    # the transform is (x - 100)/800
    norm = PercentileNormalizer(pmin=10, pmax=90, dtype=np.float32, clip=False)
    assert norm.normalize(a).max() == 1.125

    norm = PercentileNormalizer(pmin=2, pmax=99.8, dtype=np.float32, clip=True)
    assert norm.normalize(a).max() == 1.0


def gpuinfo(gpuid):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q', '-i', str(gpuid), '-d',
                           'MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    return out_dict


def getfreegpumem(id):
    print(gpuinfo(id))
    return int(gpuinfo(id)['Free'].replace('MiB', '').strip())


def getbestgpu():
    freememlist = []
    gpu_amount = int(gpuinfo(0)['Attached GPUs'])
    print("%d GPU in total." % gpu_amount)
    for id in range(gpu_amount):
        freemem = getfreegpumem(id)
        print("GPU device %d has %d MiB left." % (id, freemem))
        freememlist.append(freemem)
    idbest = freememlist.index(max(freememlist))
    print("--> GPU device %d was chosen" % idbest)
    return idbest


def get_args():
    global args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_files",
                        help="configuration file for experiment.",
                        type=str,
                        nargs='+')
    parser.add_argument("--device",
                        help="cuda device",
                        type=str,
                        required=True)
    args = parser.parse_args()
