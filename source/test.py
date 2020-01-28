import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
import utils.configs as config
import utils.util as util
import utils.dataset as ds
from denoisers import get_denoiser
from updaters import get_updater
from utils.end2end import End2end
import torch
from PIL import Image
import os
import numpy as np


def test_e2e(label, phi, cfg):
    y = label.mul(phi).sum(1)
    # util.show_tensors(y)
    with torch.no_grad():
        rec = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
        model = End2end(phi, cfg.phase, cfg.step_size, cfg.u_name, cfg.d_name)
        model.load_state_dict(torch.load(cfg.restore))
        model.eval()
        rec = model(rec, y)
        return rec


def test_iterative(label, phi, cfg):
    y = label.mul(phi).sum(1)
    rec = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
    with torch.no_grad():
        denoiser = get_denoiser(cfg.d_name, cfg.frame)
        denoiser.load_state_dict(torch.load(cfg.restore))
        denoiser.eval()
        updater = get_updater(cfg.u_name, phi, denoiser, cfg.step_size)
        params = [rec, y]
        params.extend(updater.initialize())
        for sp in range(cfg.steps):
            params = updater(*params)
            print("sp ", sp, "PSNR ", compare_psnr(label.numpy(),
                                                   np.clip(params[0].detach().cpu().numpy(), 0, 1)))
        return params[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tester Parameters",
        prog="python ./test.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--e2e', dest='tester', const=test_e2e, default=test_iterative,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)  # path
    parser.add_argument('--manual', default=False)
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='sparse')
    parser.add_argument('--l_name', default='mse')
    parser.add_argument('--group', default=4)
    parser.add_argument('--frame', default=8)
    parser.add_argument('--pixel', default=256)
    parser.add_argument('--phase', default=5)
    parser.add_argument('--steps', default=20)  # ite
    parser.add_argument('--step_size', default=0.001)  # ite
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    _, test_file, mask_file, para_dir, recon_dir, model_dir = config.general(
        args.name)

    label, phi = ds.load_test_data(test_file, mask_file)
    # util.show_tensors(label)

    start = time()
    reconstruction = args.tester(label, phi, args)
    end = time()

    reconstruction = np.clip(reconstruction.detach().cpu().numpy(), 0, 1)
    label = label.numpy()
    psnr = compare_psnr(label, reconstruction)
    t = end - start
    print("PSNR {},Time: {}".format(psnr, t))

    recon_dir = "{}/{}".format(recon_dir, int(time()))
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    for i in range(args.group):
        for j in range(args.frame):
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
