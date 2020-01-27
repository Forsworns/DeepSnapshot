# self surpvised
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers import get_denoiser
import utils.configs as cfg
import utils.util as util
import utils.dataset as ds
from utils.end2end import End2end
from torch.utils.data import DataLoader
import torch


def train_e2e(label, phi, cfg):
    torch.manual_seed(int(time()) % 10)

    model = End2end(phi, cfg.phase, cfg.step_size, cfg.u_name, cfg.d_name)
    optimizer = util.get_optimizer(o_name, model, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        model.train()
        label, y = batch
        rec = y.repeat(args.frame, 1, 1, 1).permute(
            1, 0, 2, 3).mul(phi).div(phi.sum(0)+0.0001)
        net_output = model(rec, y)
        loss = loss_func(net_output, rec)
        print("ep ", ep, "loss ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            model.eval()
            net_output = model(rec, y)
            val_loss = loss_function(net_output, label)
            val_losses.append(val_loss.item())
            
            print("ep ", ep, "loss ", loss.item(),"val loss ",val_loss.item(),"time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[
                                   0, 0], 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(),denoised)
                best_images.append(denoised)
                print("PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    rec = model(rec)
    return rec, model


def train_denoiser(label, phi, cfg):
    dataset = ds.NoisyDataset(label)

    torch.manual_seed(int(time()) % 10)

    denoiser = get_denoiser(cfg.d_name, cfg.frame)
    optimizer = util.get_optimizer(cfg.o_name, denoiser, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        denoiser.train()
        label, noisy = batch
        net_output = denoiser(noisy)
        loss = loss_func(net_output, label)
        print("ep ", ep, "loss ", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            denoiser.eval()
            net_output = denoiser(noisy)
            val_loss = loss_function(net_output, label)
            val_losses.append(val_loss.item())
            
            print("ep ", ep, "loss ", loss.item(),"val loss ",val_loss.item(),"time ", time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[
                                   0, 0], 0, 1).astype(np.float64)
                best_psnr = compare_psnr(label.numpy(),denoised)
                best_images.append(denoised)
                print("PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    rec = model(rec)
    return rec, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_gpu', default=False)    
    parser.add_argument('--device', default=None)
    parser.add_argument('--e2e', dest='trainer', const=test_e2e, default=test_iterative,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)  # e2e
    parser.add_argument('--manual', default=False)
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='sparse')
    parser.add_argument('--o_name', default='adam')
    parser.add_argument('--l_name', default='mse')
    parser.add_argument('--group', default=4)
    parser.add_argument('--frame', default=8)
    parser.add_argument('--pixel', default=256)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--epoch', default=30)
    parser.add_Argument('--batch', default=2)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--phase', default=5)  # e2e
    parser.add_argument('--steps', default=20)  # ite
    parser.add_argument('--step_size', default=0.001)  # ite
    args = parser.parse_args()

    if args.use_gpu:
        if args.device is None:
            args.device = util.getbestgpu()
    else:
        args.device = 'cpu'

    train_file, _, mask_file, para_dir, recon_dir, model_dir = cfg.general(
        args.name)
    label, phi = ds.load_train_data(train_file, mask_file)

    start = time()
    reconstruction, model = args.trainer(label, phi, args)
    end = time()
    t = end - start
    print("Training Time: {}".format(psnr, ssim, t))

    util.save_model(model, model_dir, psnr)

    config_log = config.ConfigLog(args)
    config_log.dump(para_dir)
