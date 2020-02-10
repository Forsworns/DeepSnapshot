import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers import get_denoiser
import utils.configs as config
import utils.util as util
import utils.dataset as ds
from utils.end2end import End2end
from torch.utils.data import DataLoader
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


def train_e2e(label, phi, t_label, t_phi, cfg):
    writer = SummaryWriter()
    dataset = ds.SnapshotDataset(phi, label)
    torch.manual_seed(int(time()) % 10)

    model = End2end(phi, cfg.phase, cfg.u_name, cfg.d_name, cfg.share)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    with writer as w:
        dummy_x = torch.zeros_like(label[0].unsqueeze(0))
        dummy_y = torch.zeros_like(label[0, 0].unsqueeze(0))
        w.add_graph(model, (dummy_x, dummy_y, phi))

    losses = []
    val_losses = []
    best_val_loss = 1
    best_psnr = 0

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)
        for ep_i, batch in enumerate(data_loader):
            model.train()
            label, y = batch
            rec = y.repeat(args.frame, 1, 1, 1).permute(
                1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
            rec = rec.to(cfg.device)
            y = y.to(cfg.device)
            net_output = model(rec, y, phi)

            loss = loss_func(net_output, label)/accumulation_steps
            loss.backward()
            if ep_i % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
            if (ep_i+1)%accumulation_steps ==0:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            losses.append(loss.item())
            model.eval()
            net_output = model(rec, y, phi)
            val_loss = loss_func(net_output, label)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep_i ", ep_i, "loss ", loss.item(), "val loss ",
                  val_loss, "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))

    dataset = ds.SnapshotDataset(t_phi, t_label)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    label, y = next(enumerate(data_loader))
    rec = y.repeat(args.frame, 1, 1, 1).permute(
        1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
    net_output = model(rec, y, phi)
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output.detach().cpu().numpy(), 0, 1).astype(np.float64))
    return model, best_psnr


def train_denoiser(label, phi, t_label, t_phi, cfg):
    dataset = ds.NoisyDataset(label)

    torch.manual_seed(int(time()) % 10)

    denoiser = get_denoiser(cfg.d_name, cfg.frame)
    denoiser = denoiser.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, denoiser, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    losses = []
    val_losses = []
    best_val_loss = 1

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)
        for ep_i, batch in enumerate(data_loader):
            denoiser.train()
            label, noisy = batch
            noisy = noisy.to(cfg.device)
            net_output = denoiser(noisy)
            loss = loss_func(net_output, label)/accumulation_steps
            loss.backward()
            if ep_i % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
            if (ep_i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            losses.append(loss.item())
            denoiser.eval()
            net_output = denoiser(noisy)
            val_loss = loss_func(net_output, label)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep_i ", ep_i, "loss ", loss.item(), "val loss ",
                  val_loss, "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
                print("PSNR: ", np.round(best_psnr, 2))

    dataset = ds.NoisyDataset(t_label)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    label, noisy = next(enumerate(data_loader))
    net_output = model(noisy)
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output.detach().cpu().numpy(), 0, 1).astype(np.float64))
    return denoiser, psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--device', default=None)
    parser.add_argument('--denoise', dest='trainer', const=train_denoiser, default=train_e2e,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Traffic')
    parser.add_argument('--restore', default=None)
    parser.add_argument('--manual', default=False)
    parser.add_argument('--u_name', default='plain')
    parser.add_argument('--d_name', default='sparse')
    parser.add_argument('--o_name', default='adam')
    parser.add_argument('--l_name', default='mse')
    parser.add_argument('--group', type=int, default=4)
    parser.add_argument('--frame', type=int, default=8)
    parser.add_argument('--pixel', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--phase', type=int, default=2)
    parser.add_argument('--share', type=bool, default=False)
    parser.add_argument('--poor', type=int, default=1)
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    train_file, test_file, mask_file, para_dir, recon_dir, model_dir = config.general(
        args.name)
    t_label, t_phi = ds.load_test_data(test_file, mask_file, False)
    label, phi = ds.load_train_data(train_file, mask_file, False)
    print(label.shape)

    start = time()
    model, psnr = args.trainer(label, phi, t_label, t_phi, args)
    end = time()
    t = end - start
    print("PSNR {}, Training Time: {}".format(psnr, t))

    util.save_model(model, model_dir, psnr)
    args_dict = vars(args)
    args_dict.pop('trainer')
    config_log = config.ConfigLog(args_dict)
    config_log.dump(para_dir)
