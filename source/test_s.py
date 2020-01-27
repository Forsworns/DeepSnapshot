# self surpvised
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
import utils.configs as cfg
import utils.util as util
import utils.dataset as ds
from denoisers import get_denoiser
from utils.end2end import End2end
import torch


def test_e2e(label, phi, cfg):
    y = label.mul(phi).sum(1)
    # util.show_tensors(y)
    with torch.no_grad():
        rec = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
        model = End2end(phi, cfg.phase, cfg.u_name,
                        cfg.d_name, cfg.step_size)
        model.load_state_dict(cfg, restore)
        model.eval()
        rec = model(rec, y)
        return rec


def test_iterative(label, phi, cfg):
    y = label.mul(phi).sum(1)
    with torch.no_grad():
        denoiser = get_denoiser(cfg.d_name, cfg.frame)
        denoier.load_state_dict(cfg.restore)
        denoiser.eval()
        updater = get_updater(cfg.u_name, phi, y, denoiser, cfg.step_size)
        params = [y]
        params.extend(updater.initialize())
        for sp in range(self.steps):
            params = updater(params)
            print("sp ", sp, "PSNR ", compare_psnr(label,params[0]))
        return params[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-supervised Tester Parameters",
        prog="python ./test_s.py",
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
    parser.add_argument('--steps', default=20)  # ite
    parser.add_argument('--step_size', default=0.001)  # ite
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    _, test_file, mask_file, para_dir, recon_dir, model_dir = cfg.general(
        args.name)

    label, phi = ds.load_test_data(test_file, mask_file)
    # util.show_tensors(label)

    start = time()
    reconstruction = args.tester(label, phi, y, args)
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
