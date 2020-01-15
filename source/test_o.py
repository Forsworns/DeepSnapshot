# one shot, different the other testcases, need to train during test
import argparse
from time import time
from skimage.measure import compare_ssim, compare_psnr
from denoisers.denoiers import get_denoiser
from updaters.updaters import get_updater
import utils.configs as cfg
import utils.util

FRAME = 8
VALIDATE_AFTER = 10
TRAIN_VLIDATE_SPLIT = 0.9


def test_e2e(x,y,phi,u_name,d_name,u_cfg,d_cfg):
    pass

def test_iterative(x,y,phi,u_name,d_name,u_cfg,d_cfg):
    for sp in xrange(u_cfg['steps']):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters", 
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--e2e', dest='trainer', const=train_e2e, default=train_denoiser, action='store_const', help="train a denoiser or reconstruction model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--u_name', default='fista')
    parser.add_argument('--d_name', default='dncnn')
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--epoch', default=500)
    parser.add_argument('--phase',default=5) # e2e
    parser.add_argument('--steps', default=20)
    parser.add_argument('--step_size', default=0.1)
    args = parser.parse_args()

    train_file, test_file, mask_file, para_dir, recon_dir, model_dir = cfg.general(args.name)


    u_cfg = {}
    d_cfg = {}
    config = cfg.Config(args.u_name,args.d_name,u_cfg,d_cfg)

    show_tensor(x)

    args.trainer(x,y,phi,u_name,d_name,u_cfg,d_cfg)
