# one shot, different the other testcases, need to train during test
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers import get_denoiser
from updaters import get_updater
import utils.configs as cfg
import utils.util as util
import utils.dataset as ds
from utils.mask import Masker
from end2end import End2end


def test_e2e(x, y, phi, cfg):
    torch.manual_seed(int(time()) % 10)
    rec = phi.mul(x)
    if cfg.restore is None:
        model = End2end(cfg.depth, cfg.u_name,
                        cfg.d_name, cfg.frame, cfg.frame)
        optimizer = util.get_optimizer(o_name, model, cfg.learning_rate)
        loss_func = util.get_loss(cfg.l_name)
        masker = Masker(width=4, mode='interpolate')

        losses = []
        val_losses = []
        best_images = []
        best_val_loss = 1

        for ep in xrange(cfg.epoch):
            net_input, mask = masker.mask(rec, i % (masker.n_masks - 1))
            net_output = model(net_input)
            loss = loss_func(net_output*mask, rec*mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 10 == 0:
                losses.append(loss.item())
                model.eval()
                net_input, mask = masker.mask(
                    rec, (i+1) % (masker.n_masks - 1))
                net_output = model(net_input)
                val_loss = loss_function(net_output*mask, rec*mask)
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
    else:
        model = load_model(cfg.restore)
    rec = model(rec)
    return rec, model

# the denoiser is not blind, so maybe not suitable to save/restore a denoiser?


def test_iterative(x, y, phi, cfg):
    torch.manual_seed(int(time()) % 10)
    rec = phi.mul(x)
    denoiser = get_denoiser(cfg.d_name)
    updater = get_updater(cfg.u_name)
    optimizer = util.get_optimizer(o_name, model, cfg.learning_rate)
    loss_func = util.get_loss(cfg.l_name)
    masker = Masker(width=4, mode='interpolate')

    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    for sp in xrange(cfg.steps):
        rec = updater(rec)
        for ep in xrange(cfg.epoch):
            net_input, mask = masker.mask(rec, i % (masker.n_masks - 1))
            net_output = denoiser(net_input)
            loss = loss_func(net_output*mask, rec*mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ep % 10 == 0:
                losses.append(loss.item())
                denoiser.eval()
                net_input, mask = masker.mask(
                    rec, (i+1) % (masker.n_masks - 1))
                net_output = denoiser(net_input)
                val_loss = loss_function(net_output*mask, rec*mask)
                val_losses.append(val_loss.item())
                print("(", sp, "-", ep, ") Loss: \t", round(loss.item(), 5),
                      "\tVal Loss: \t", round(val_loss.item(), 5), "Time:\t", time.time())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    net_output = np.clip(model(rec).detach().cpu().numpy()[
                                         0, 0], 0, 1).astype(np.float64)
                    best_psnr = compare_psnr(net_output, image)
                    best_images.append(net_output)
                    print("\tModel PSNR: ", np.round(best_psnr, 2))
        rec = denoiser(rec)
    return rec, denoiser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters",
        prog="python ./test_o.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--e2e', dest='tester', const=test_e2e, default=test_iterative,
                        action='store_const', help="test a iterative method or end2end model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--restore', default=None)  # e2e
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='dncnn')
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
    print(label.size(),phi.size())
    util.show_tensor(label)

    y = np.sum(np.multiply(label, phi), axis=3)
    x = np.tile(np.reshape(np.multiply(yinput, sumPhi),
                           [-1, pixel, pixel, 1]), [1, 1, 1, nFrame])
    y = np.reshape(yinput, (-1, pixel, pixel, 1))

    start = time()
    reconstruction, model = args.tester(x, y, phi, u_name, d_name, args)
    end = time()

    psnr = compare_psnr(label, reconstruction)
    ssim = compare_ssim(label, reconstruction)
    t = end - start
    print("PSNR {}, SSIM: {},Time: {}".format(psnr, ssim, t))

    save_model(model, model_dir, psnr)

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
