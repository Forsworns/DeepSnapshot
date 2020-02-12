# self surpvised
import argparse
from time import time

import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import utils.configs as config
import utils.dataset as ds
import utils.util as util
from denoisers import get_denoiser
from utils.end2end import End2end
from utils.mask import Masker


def train_e2e(label, phi, t_label, t_phi, cfg):
    # writer = SummaryWriter()
    dataset = ds.SnapshotDataset(phi, label)
    torch.manual_seed(int(time()) % 10)

    phi = phi.to(cfg.device)
    model = End2end(phi, cfg)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    # with writer as w:
    #     dummy_x = torch.zeros_like(label[0].unsqueeze(0))
    #     dummy_y = torch.zeros_like(label[0, 0].unsqueeze(0))
    #     w.add_graph(model, (dummy_x, dummy_y, phi))

    losses = []
    val_losses = []
    best_val_loss = 1

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(
            dataset, batch_size=cfg.batch, shuffle=True, drop_last=True)
        optimizer.zero_grad()
        last_batch = len(data_loader) // cfg.batch - 1
        for ep_i, batch in enumerate(data_loader):
            label, y = batch
            rec = y.repeat(args.frame, 1, 1, 1).permute(
                1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
            rec = rec.to(cfg.device)
            y = y.to(cfg.device)
            label = label.to(cfg.device)
            if ep_i == last_batch:
                break
            net_input, mask = masker.mask(rec, ep_i % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            model.train()
            net_output = model(net_input, y, phi)
            loss = loss_func(net_output*mask, noisy*mask)/accumulation_steps
            loss.backward()
            if (ep_i+1) % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            losses.append(loss.item())
            model.eval()
            net_input, mask = masker.mask(rec, (ep_i+1) % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = model(net_input, y, phi)
            val_loss = loss_func(net_output*mask, rec*mask)
            scheduler.step(val_loss)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep_i ", ep, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "lr", optimizer.param_groups[0]['lr'], "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))
                util.save(model, best_psnr, best_img, label.cpu().numpy(), cfg)

    dataset = ds.SnapshotDataset(t_phi, t_label)
    t_phi = t_phi.to(cfg.device)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    label, y = next(iter(data_loader))
    rec = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(t_phi.cpu()).div(t_phi.cpu().sum(0)+0.0001)
    rec = rec.to(cfg.device)
    y = y.to(cfg.device)
    net_output = model(rec, y, phi).detach().cpu().numpy()
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output, 0, 1).astype(np.float64))
    return model, psnr, net_output


def train_denoiser(label, phi, t_label, t_phi, cfg):
    dataset = ds.NoisyDataset(label)
    torch.manual_seed(int(time()) % 10)

    denoiser = get_denoiser(cfg.d_name, cfg.frame)
    denoiser = denoiser.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, denoiser, cfg.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    losses = []
    val_losses = []
    best_val_loss = 1

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(
            dataset, batch_size=cfg.batch, shuffle=True, drop_last=True)
        optimizer.zero_grad()
        last_batch = len(data_loader) // cfg.batch - 1
        for ep_i, batch in enumerate(data_loader):
            label, noisy = batch
            noisy = noisy.to(cfg.device)
            label = label.to(cfg.device)
            if ep_i == last_batch:
                break
            net_input, mask = masker.mask(noisy, ep_i % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            denoiser.train()
            net_output = denoiser(net_input)
            loss = loss_func(net_output*mask, noisy*mask)/accumulation_steps
            loss.backward()
            if (ep_i+1) % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            losses.append(loss.item())
            denoiser.eval()
            net_input, mask = masker.mask(
                noisy, (ep_i+1) % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = denoiser(net_input)
            val_loss = loss_func(net_output*mask, noisy*mask)
            scheduler.step(val_loss)
            val_loss = val_loss.item()
            val_losses.append(val_loss)
            print("ep ", ep, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "lr", optimizer.param_groups[0]['lr'], "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.cpu().numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))

    dataset = ds.NoisyDataset(t_label)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    label, noisy = next(enumerate(data_loader))
    net_output = model(noisy).detach().cpu().numpy()
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output, 0, 1).astype(np.float64))
    return denoiser, psnr, net_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-supervised Trainer Parameters",
        prog="python ./train_s.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--denoise', dest='trainer', const=train_denoiser, default=train_e2e,
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
    parser.add_argument('--phase', type=int, default=1)
    parser.add_argument('--share', type=bool, default=False)
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    train_file, test_file, mask_file, para_dir, recon_dir, model_dir = config.general(
        args.name)
    label, phi = ds.load_train_data(train_file, mask_file, True)
    t_label, t_phi = ds.load_train_data(test_file, mask_file, True)
    print(label.shape)
    start = time()
    model, psnr, reconstruction = args.trainer(
        label, phi, t_label, t_phi, args)
    end = time()
    t = end - start
    print("PSNR {}, Training Time: {}".format(psnr, t))

    util.save(model, psnr, reconstruction, t_label.cpu().numpy(), args)
