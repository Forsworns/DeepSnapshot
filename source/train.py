# self surpvised
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers.denoiers import get_denoiser
import utils.configs as cfg
import utils.util as util
import utils.dataset as ds
from end2end import End2end
from torch.utils.data import Dataloader


def train_e2e(dataset, phi, cfg):
    torch.manual_seed(int(time()) % 10)

    sumPhi = phi.mul(phi.T)
    model = End2end(cfg.depth, cfg.u_name, cfg.d_name, cfg.frame, cfg.frame)
    optimizer = util.get_optimizer(o_name, model, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        label, y = batch
        rec = np.tile(np.reshape(np.multiply(y, sumPhi),
                                 [-1, pixel, pixel, 1]), [1, 1, 1, args.frame])
        net_output = model(rec)
        loss = loss_func(net_output, rec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            model.eval()
            net_output = model(rec)
            val_loss = loss_function(net_output, label)
            val_losses.append(val_loss.item())
            print("(", sp, "-", ep, ") Loss: \t", round(loss.item(), 5),
                  "\tVal Loss: \t", round(val_loss.item(), 5), "Time:\t", time.time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[
                                   0, 0], 0, 1).astype(np.float64)
                best_psnr = compare_psnr(denoised, image)
                best_images.append(denoised)
                print("\tModel PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    rec = model(rec)
    return rec, model


def train_denoiser(dataset, cfg):
    torch.manual_seed(int(time()) % 10)

    denoiser = get_denoiser(d_name)
    optimizer = util.get_optimizer(o_name, denoiser, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    data_loader = DataLoader(dataset, batch_size=cfg.batch, shuffle=True)

    for ep, batch in enumerate(data_loader):
        label, noisy = batch
        net_output = model(noisy)
        loss = loss_func(net_output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            losses.append(loss.item())
            model.eval()
            net_output = model(noisy)
            val_loss = loss_function(net_output, label)
            val_losses.append(val_loss.item())
            print("(", sp, "-", ep, ") Loss: \t", round(loss.item(), 5),
                  "\tVal Loss: \t", round(val_loss.item(), 5), "Time:\t", time.time())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[
                                   0, 0], 0, 1).astype(np.float64)
                best_psnr = compare_psnr(denoised, image)
                best_images.append(denoised)
                print("\tModel PSNR: ", np.round(best_psnr, 2))
        if ep == cfg.epoch:
            break

    rec = model(rec)
    return rec, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--e2e', default=False,
                        help="train a denoiser or reconstruction model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='dncnn')
    parser.add_argument('--frame', default=8)
    parser.add_argument('--total_frame', default=32)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--epoch', default=100)
    parser.add_Argument('--batch', default=32)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--tv_split', default=0.9)
    parser.add_argument('--phase', default=5)  # e2e
    args = parser.parse_args()

    train_file, _, mask_file, para_dir, recon_dir, model_dir = cfg.general(
        args.name)
    label, phi = ds.load_train_data(train_file, mask_file)

    start = time()
    if args.e2e:
        dataset = ds.SnapshotDataset(phi, label)
        reconstruction, model = train_e2e(dataset, phi, args)
    else:
        dataset = ds.NoisyDataset(label)
        reconstruction, model = train_denoiser(dataset, args)
    end = time()

    psnr = compare_psnr(label, reconstruction)
    ssim = compare_ssim(label, reconstruction)
    t = end - start
    print("PSNR {}, SSIM: {},Training Time: {}".format(psnr, ssim, t))

    util.save_model(model, model_dir, psnr)

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
