import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
import utils.configs as cfg
import utils.util as util
from denoisers.denoisers import get_denoiser
from iterative import Iterative
from end2end import End2end
import torch


def test_e2e(y, cfg):
    with torch.no_grad():
        rec = phi.mul(x)
        model = End2end(cfg.phase, cfg.u_name,
                        cfg.d_name, cfg.frame, cfg.frame)
        model.load_state_dict(cfg, restore)
        rec = model(y)
        return rec


def test_iterative(y, cfg):
    with torch.no_grad():
        denoiser = get_denoiser(cfg.d_name)
        denoier.load_state_dict(cfg.restore)
        model = Iterative(cfg.steps, cfg.step_size, cfg.u_name, denoiser)
        rec = model(y)
        return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--e2e', dest='tester', const=test_e2e, default=test_denoiser,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='dncnn')
    parser.add_argument('--frame', default=8)
    parser.add_argument('--total_frame', default=32)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--phase', default=5)  # e2e
    parser.add_argument('--steps', default=20)  # ite
    parser.add_argument('--step_size', default=0.1)  # ite
    args = parser.parse_args()

    _, test_file, mask_file, para_dir, recon_dir, model_dir = cfg.general(
        args.name)

    label, phi = ds.load_test_data(test_file, mask_file)
    show_tensor(label)

    y = np.sum(np.multiply(label, phi), axis=3)
    x = np.tile(np.reshape(np.multiply(yinput, sumPhi),
                           [-1, pixel, pixel, 1]), [1, 1, 1, args.frame])
    y = np.reshape(yinput, (-1, pixel, pixel, 1))

    start = time()
    reconstruction = args.tester(x, y, phi, u_name, d_name, args)
    end = time()

    psnr = compare_psnr(label, reconstruction)
    ssim = compare_ssim(label, reconstruction)
    t = end - start
    print("PSNR {}, SSIM: {},Time: {}".format(psnr, ssim, t))

    cfg = cfg.ConfigLog(args)
    cfg.dump(para_dir)

    for i in range(args.total_frame/args.frame):
        for j in range(args.frame):
            PSNR = psnr(rec[i, :, :, j], xoutput[i, :, :, j])
            print("Frame %d, PSNR: %.2f" % (i*args.frame+j, PSNR))
            outImg = np.hstack((xoutput[i, :, :, j], rec[i, :, :, j]))
            imgRecName = "%s/frame%d_PSNR%.2f.png" % (
                recon_dir, i*args.frame+j, PSNR)
            imgRec = Image.fromarray(
                np.clip(255*outImg, 0, 255).astype(np.uint8))
            imgRec.save(imgRecName)
