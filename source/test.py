import argparse
import os
from time import time

import numpy as np
import torch
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

import utils.configs as config
import utils.dataset as ds
import utils.util as util
from denoisers import get_denoiser
from updaters import get_updater
from utils.end2end import End2end


def test_e2e(label, phi, cfg):
    y = label.mul(phi).sum(1)
    # util.show_tensors(y)
    phi = phi.to(cfg.device)
    label = label.to(cfg.device)
    y = y.to(cfg.device)
    with torch.no_grad():
        initial = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
        initial.to(cfg.device)
        model = End2end(phi, cfg)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        model.load_state_dict(torch.load(cfg.restore))
        model.eval()
        initial = model(initial, y, phi)
        return initial

'''
def test_iterative(label, phi, cfg):
    y = label.mul(phi).sum(1)
    initial = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
    phi = phi.to(cfg.device)
    initial = initial.to(cfg.device)
    y = y.to(cfg.device)
    with torch.no_grad():
        denoiser = get_denoiser(cfg.d_name, cfg.frame)
        denoiser.load_state_dict(torch.load(cfg.restore))
        denoiser.eval()
        denoiser.to(cfg.device)
        updater = get_updater(cfg.u_name, phi, denoiser, cfg.step_size)
        updater.to(cfg.device)
        params = [initial, y]
        params.extend(updater.initialize())
        for sp in range(cfg.steps):
            params = updater(*params)
            print("sp ", sp, "PSNR ", compare_psnr(label.numpy(),
                                                   np.clip(params[0].detach().cpu().numpy(), 0, 1)))
        return params[0]
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tester Parameters",
        prog="python ./test.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--denoise', dest='tester', const=test_iterative, default=test_e2e,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)  # path
    parser.add_argument('--manual', type=bool, default=False)
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='snet')
    parser.add_argument('--phase', type=int, default=5)
    parser.add_argument('--group', type=int, default=4)
    parser.add_argument('--frame', type=int, default=8)
    parser.add_argument('--pixel', type=int, default=256)
    parser.add_argument('--share', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=20)  # ite
    parser.add_argument('--step_size', type=float, default=0.001)  # ite
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
