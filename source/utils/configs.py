import json
import os
from time import time


def general(name):
    pixel = 256
    root_dir = os.path.dirname(__file__) + '/../../'
    train_file = root_dir + 'data/train/train%s%d.mat' % (name, pixel)
    test_file = root_dir + 'data/test/test%s%d.mat' % (name, pixel)
    mask_file = root_dir + 'data/mask%d.mat' % (pixel)
    para_dir = root_dir + 'results/parameter/Para%s%d/%d' % (name, pixel)
    recon_dir = root_dir + 'results/reconstruction/Img%s%d/%d' % (name, pixel,int(time()))
    model_dir = root_dir + 'results/model/Model%s%d/%d' % (name, pixel,int(time()))
    if not os.path.exists(para_dir):
        os.makedirs(para_dir)
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return train_file, test_file, mask_file, para_dir, recon_dir, model_dir


class ConfigLog(object):
    def __init__(self, cfg, psnr=None, ssim=None):
        self.cfg = cfg
        if psnr is not None:
            self.cfg.update({'psnr': psnr})
        if ssim is not None:
            self.cfg.update({'ssim': ssim})

    def dump(self, para_dir):
        t = int(time())
        cfg_file = "{}/{}.json".format(para_dir, t)
        with open(cfg_file, "w") as f:
            json.dump(self.cfg, f)

    def load(self, cfg_file):
        try:
            with open(d_cfg_file, "r") as f:
                self.cfg = json.load(f)
        except FileNotFoundError:
            print("Error! Cannot find cfg for " + cfg_file)
