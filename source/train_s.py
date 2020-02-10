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
from torch.optim import lr_scheduler


def train_e2e(label, phi, t_label, t_phi, cfg):
    dataset = ds.SnapshotDataset(phi, label)
    torch.manual_seed(int(time()) % 10)

    model = End2end(phi, cfg.phase, cfg.u_name, cfg.d_name, cfg.share)
    model = model.to(cfg.device)
    optimizer = util.get_optimizer(cfg.o_name, model, cfg.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    losses = []
    val_losses = []
    best_val_loss = 1

    accumulation_steps = cfg.poor
    for ep in range(cfg.epoch):
        data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True, drop_last=True, num_workers=4)
        for ep_i, batch in enumerate(data_loader):
            model.train()
            label, y = batch
            rec = y.repeat(args.frame, 1, 1, 1).permute(
                1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
            y = y.to(cfg.device)
            net_input, mask = masker.mask(rec, ep_i % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = model(net_input, y, phi)
            loss = loss_func(net_output*mask, noisy*mask)/accumulation_steps
            loss.backward()
            if ep_i % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
            if (ep_i+1) % accumulation_steps == 0:
                scheduler.step(loss)
                optimizer.step()
                optimizer.zero_grad()
        with torch.no_grad():
            losses.append(loss.item())
            model.eval()
            net_input, mask = masker.mask(rec, (ep_i+1) % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = model(net_input, y, phi)
            val_loss = loss_func(net_output*mask, rec*mask)
            val_loss = val_loss.item()
            val_losses.append(val_loss)

            print("ep_i ", ep_i, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "time ", time())

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
    net_output = model(rec, y, phi).detach().cpu().numpy()
    psnr = compare_psnr(label.numpy(), np.clip(
        net_output, 0, 1).astype(np.float64))
    return denoiser, psnr, net_output


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
        data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True, drop_last=True, num_workers=4)
        for ep_i, batch in enumerate(data_loader):
            denoiser.train()
            label, noisy = batch
            net_input, mask = masker.mask(noisy, ep_i % (masker.n_masks - 1))
            net_input = net_input.to(cfg.device)
            net_output = denoiser(net_input)
            loss = loss_func(net_output*mask, noisy*mask)/accumulation_steps
            loss.backward()
            if ep_i % accumulation_steps == 0:
                print("ep", ep, "ep_i ", ep_i, "loss ", loss.item())
            if (ep_i+1) % accumulation_steps == 0:
                scheduler.step(loss)
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
            val_loss = val_loss.item()
            val_losses.append(val_loss)
            print("ep_i ", ep_i, "loss ", round(loss.item(), 5), "val loss ",
                  round(val_loss, 5), "time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_img = np.clip(
                    net_output.detach().cpu().numpy(), 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(), best_img)
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

    start = time()
    model, psnr, reconstruction = args.trainer(label, phi, t_label, t_phi, args)
    end = time()
    t = end - start
    print("PSNR {}, Training Time: {}".format(psnr, t))

    util.save_model(model, model_dir, psnr)
    args_dict = vars(args)
    args_dict.pop('trainer')
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
