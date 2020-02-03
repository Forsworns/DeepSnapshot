# self surpvised
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers import get_denoiser
import utils.configs as config
import utils.util as util
import utils.dataset as ds
from utils.mask import Masker
from utils.end2end import End2end
from torch.utils.data import DataLoader
import torch
import numpy as np


def train_e2e(label, phi, cfg):
    dataset = ds.SnapshotDataset(phi, label)
    torch.manual_seed(int(time()) % 10)

    model = End2end(phi, cfg.phase, cfg.u_name, cfg.d_name)
    model = model.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    losses = []
    val_losses = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        model.train()
        label, y = batch
        rec = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
        y = y.to(cfg.device)
        net_input, mask = masker.mask(rec, ep % (masker.n_masks - 1))
        net_input = net_input.to(cfg.device)
        net_output = model(net_input, y)
        loss = loss_func(net_output*mask, rec*mask)
        print("ep ", ep, "loss ", round(loss.item(), 5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            model.eval()
            net_input, mask = masker.mask(rec, (ep+1) % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = model(net_input, y)
            val_loss = loss_func(net_output*mask, rec*mask)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep ", ep, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(model(rec, y).detach(
                ).cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    return model, best_psnr


def train_denoiser(label, phi, cfg):
    dataset = ds.NoisyDataset(label)
    torch.manual_seed(int(time()) % 10)

    denoiser = get_denoiser(cfg.d_name, cfg.frame)
    denoiser = denoiser.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, denoiser, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    losses = []
    val_losses = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        denoiser.train()
        label, noisy = batch
        net_input, mask = masker.mask(noisy, ep % (masker.n_masks - 1))
        net_input = net_input.to(cfg.device)
        net_output = denoiser(net_input)
        loss = loss_func(net_output*mask, noisy*mask)
        print("ep ", ep, "loss ", round(loss.item(), 5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            denoiser.eval()
            net_input, mask = masker.mask(noisy, (ep+1) % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = denoiser(net_input)
            val_loss = loss_func(net_output*mask, noisy*mask)
            val_loss = val_loss.item()
            val_losses.append(val_loss)
            print("ep ", ep, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(denoiser(noisy).detach(
                ).cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    return denoiser, best_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-supervised Trainer Parameters",
        prog="python ./train_s.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--e2e', dest='trainer', const=train_e2e, default=train_denoiser,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--manual', type=bool, default=False)
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='sparse')
    parser.add_argument('--o_name', default='adam')
    parser.add_argument('--l_name', default='mse')
    parser.add_argument('--group', type=int, default=4)
    parser.add_argument('--frame', type=int, default=8)
    parser.add_argument('--pixel', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--phase', type=int, default=1)
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    train_file, _, mask_file, para_dir, recon_dir, model_dir = config.general(
        args.name)
    label, phi = ds.load_train_data(train_file, mask_file)

    start = time()
    model, psnr = args.trainer(label, phi, args)
    end = time()
    t = end - start
    print("PSNR {}, Training Time: {}".format(psnr, t))

    util.save_model(model, model_dir, psnr)
    args_dict = vars(args)
    args_dict.pop('trainer')
    config_log = config.ConfigLog(args_dict)
    config_log.dump(para_dir)
