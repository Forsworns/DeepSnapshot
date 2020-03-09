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
from losses import get_loss
from utils.end2end import End2end


def train(label, phi, t_label, t_phi, cfg):
    # writer = SummaryWriter()
    dataset = ds.SnapshotDataset(phi, label)
    t_dataset = ds.SnapshotDataset(t_phi, t_label)

    phi = phi.to(cfg.device)
    model = End2end(phi, cfg)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 0.5, cfg.scheduler)
    loss_func = get_loss(cfg)

    # with writer as w:
    #     dummy_x = torch.zeros_like(label[0].unsqueeze(0))
    #     dummy_y = torch.zeros_like(label[0, 0].unsqueeze(0))
    #     w.add_graph(model, (dummy_x, dummy_y, phi))

    losses = []
    val_losses = []
    best_val_loss = 1
    best_psnr = 0

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(
            dataset, batch_size=cfg.batch, shuffle=True, drop_last=True)
        optimizer.zero_grad()

        data_loader = iter(data_loader)
        v_label, v_y = next(data_loader)
        v_initial = v_y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi.cpu()).div(phi.cpu().sum(0)+0.0001)
        v_initial = v_initial.to(cfg.device)
        v_y = v_y.to(cfg.device)
        v_label = v_label.to(cfg.device)

        for ep_i, batch in enumerate(data_loader):
            label, y = batch
            initial = y.repeat(args.frame, 1, 1, 1).permute(
                1, 0, 2, 3).mul(phi.cpu()).div(phi.cpu().sum(0)+0.0001)
            initial = initial.to(cfg.device)
            y = y.to(cfg.device)
            label = label.to(cfg.device)
            model.train()
            layers = model(initial, y, phi)
            net_output = layers[-1]
            loss = loss_func(layers, label)
            loss.backward()
            if (ep_i+1) % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            losses.append(loss.item())
            model.eval()
            v_layers = model(v_initial, v_y, phi)
            net_output = v_layers[-1]
            val_loss = loss_func(v_layers, v_label)
            scheduler.step(val_loss)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep ", ep, "loss ", loss.item(), "val loss ",
                  val_loss, "lr", optimizer.param_groups[0]['lr'], "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(v_label.cpu().numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))
                util.save(model, best_psnr, best_img, v_label.cpu().numpy(), cfg)

    t_phi = t_phi.to(cfg.device)
    data_loader = DataLoader(
        t_dataset, batch_size=t_label.shape[0], shuffle=True)
    label, y = next(iter(data_loader))
    initial = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(t_phi.cpu()).div(t_phi.cpu().sum(0)+0.0001)
    initial = initial.to(cfg.device)
    y = y.to(cfg.device)
    layers = model(initial, y, t_phi)
    net_output = layers[-1].detach().cpu().numpy()
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output, 0, 1).astype(np.float64))
    return model, psnr, net_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--name', default='Traffic')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--manual', default=False)
    parser.add_argument('--u_name', default='ista')
    parser.add_argument('--d_name', default='snet')
    parser.add_argument('--o_name', default='adam')
    parser.add_argument('--l_name', default='layer')
    parser.add_argument('--l_layer', default=0.2)
    parser.add_argument('--group', type=int, default=4)
    parser.add_argument('--frame', type=int, default=8)
    parser.add_argument('--pixel', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--phase', type=int, default=2)
    parser.add_argument('--share', type=bool, default=False)
    parser.add_argument('--poor', type=int, default=1)
    parser.add_argument('--scheduler', type=int, default=5)
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    train_file, test_file, mask_file, _, _, _ = config.general(
        args.name)
    t_label, t_phi = ds.load_test_data(test_file, mask_file, False)
    if args.name == "Traffic":
        label, phi = ds.load_train_data(train_file, mask_file, False)
    else:
        label, phi = ds.load_train_data(train_file, mask_file, True)
    print(label.shape)

    start = time()
    model, psnr, reconstruction = train(
        label, phi, t_label, t_phi, args)
    end = time()
    t = end - start
    print("PSNR {}, Training Time: {}".format(psnr, t))

    util.save(model, psnr, reconstruction, t_label.cpu().numpy(), args)
