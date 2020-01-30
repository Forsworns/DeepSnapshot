# one shot, different the other testcases, need to train during test
import argparse
from time import time
import torch
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
from denoisers import get_denoiser
from updaters import get_updater
import utils.configs as config
import utils.util as util
import utils.dataset as ds
from utils.mask import Masker
from utils.end2end import End2end
from PIL import Image
import os


def test_e2e(label, phi, cfg):
    y = label.mul(phi).sum(1)
    # util.show_tensors(y)

    torch.manual_seed(int(time()) % 10)
    if cfg.restore is None:
        model = End2end(phi, cfg.phase, cfg.step_size, cfg.u_name,
                        cfg.d_name)
        optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
        loss_func = util.get_loss(cfg.l_name)
        masker = Masker(frame=cfg.frame, width=4, mode='interpolate')

        losses = []
        val_losses = []
        best_val_loss = 1

        rec = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)

        for ep in range(cfg.epoch):
            model.train()
            net_input, mask = masker.mask(rec, ep % (masker.n_masks - 1))
            net_output = model(net_input, y)
            loss = loss_func(net_output*mask, rec*mask)
            print("ep ", ep, "loss ", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 10 == 0:
                losses.append(loss.item())
                model.eval()
                net_input, mask = masker.mask(
                    rec, (ep+1) % (masker.n_masks - 1))
                net_output = model(net_input, y)
                val_loss = loss_func(net_output*mask, rec*mask)
                val_loss = val_loss.item()
                val_losses.append(val_loss)
                print("ep ", ep, "loss ", loss.item(), "val loss ",
                      val_loss, "time ", time())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_img = np.clip(model(rec, y).detach(
                    ).cpu().numpy(), 0, 1).astype(np.float64)
                    best_psnr = compare_psnr(label.numpy(), best_img)
                    print("PSNR: ", np.round(best_psnr, 2))
    else:
        model = load_model(cfg.restore)
    rec = model(rec, y)
    return rec, model

# the denoiser is not blind, so maybe not suitable to save/restore a denoiser?


def test_iterative(label, phi, cfg):
    y = label.mul(phi).sum(1)
    # util.show_tensors(y)
    torch.manual_seed(int(time()) % 10)
    denoiser = get_denoiser(cfg.d_name, cfg.frame)
    updater = get_updater(cfg.u_name, phi, denoiser, cfg.step_size)
    optimizer = util.get_optimizer(cfg.o_name, denoiser, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='zero')

    losses = []
    val_losses = []
    best_val_loss = 1

    rec = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
    # may divide zero -> "nan"
    net_input = rec
    params = [rec, y]
    params.extend(updater.initialize())

    for sp in range(cfg.steps):
        for ep in range(cfg.epoch):
            denoiser.train()
            net_input, mask = masker.mask(rec, ep % (masker.n_masks - 1))
            net_output = denoiser(net_input)
            loss = loss_func(net_output*mask, rec*mask)
            # util.show_tensors(net_output.detach().cpu())
            print("step: ", sp, "ep ", ep, "loss ", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep % 10 == 0:
                losses.append(loss.item())
                denoiser.eval()
                net_input, mask = masker.mask(
                    rec, (ep+1) % (masker.n_masks - 1))
                net_output = denoiser(net_input)
                val_loss = loss_func(net_output*mask, rec*mask)
                val_loss = val_loss.item()
                val_losses.append(val_loss)
                print("ep ", ep, "loss ", loss.item(), "val loss ",
                      val_loss, "time ", time())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_img = np.clip(denoiser(rec).detach(
                    ).cpu().numpy(), 0, 1).astype(np.float64)
                    best_psnr = compare_psnr(label.numpy(), best_img)
                    print("PSNR: ", np.round(best_psnr, 2))

        with torch.no_grad():
            updater = get_updater(cfg.u_name, phi, denoiser, cfg.step_size)
            params = updater(*params)
            rec = params[0]
    return rec, denoiser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One shot Tester Parameters",
        prog="python ./test_o.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--e2e', dest='tester', const=test_e2e, default=test_iterative,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)  # e2e
    parser.add_argument('--manual', default=False) # manual settings of updaters
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='sparse')
    parser.add_argument('--o_name', default='adam')
    parser.add_argument('--l_name', default='mse')
    parser.add_argument('--group', default=4)
    parser.add_argument('--frame', default=8)
    parser.add_argument('--pixel', default=256)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--phase', default=2)  # e2e
    parser.add_argument('--steps', default=10)  # ite
    parser.add_argument('--step_size', default=0.001)
    args = parser.parse_args()

    # use xx.to(cfg.device) to transfer the model or data
    # to different devices
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
    reconstruction, model = args.tester(label, phi, args)
    end = time()

    reconstruction = np.clip(reconstruction.detach().cpu().numpy(), 0, 1)
    label = label.numpy()
    psnr = compare_psnr(label, reconstruction)
    t = end - start
    print("PSNR {},Time: {}".format(psnr, t))

    util.save_model(model, model_dir, psnr)

    args_dict = vars(args)
    args_dict.pop('tester')
    config_log = config.ConfigLog(args_dict)
    config_log.dump(para_dir)

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
