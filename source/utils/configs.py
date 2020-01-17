from time import time
import json
import os


def general(name):
    pixel = 256
    root_dir = os.path.dirname(__file__) + '/../../'
    train_file = root_dir + 'data/train/train%s%d.mat' % (name, pixel)
    test_file = root_dir + 'data/test/test%s%d.mat' % (name, pixel)
    mask_file = root_dir + 'data/mask%d.mat' % (pixel)
    para_dir = root_dir + 'results/parameter/Para%s%d' % (name, pixel)
    recon_dir = root_dir + 'results/reconstruction/Img%s%d' % (name, pixel)
    model_dir = root_dir + 'results/model/Model%s%d' % (name, pixel)
    if not os.path.exist(para_dir):
        os.path.makedirs(para_dir)
    if not os.path.exist(recon_dir):
        os.path.makedirs(recon_dir)
    if not os.path.exist(model_dir):
        os.path.makedirs(model_dir)
    return train_file, test_file, mask_file, para_dir, recon_dir, model_dir


class ConfigLog(object):
    def __init__(self, cfg, psnr=None, ssim=None):
        self.cfg = cfg
        if psnr is not None:
            self.cfg.update({'psnr': psnr})
        if ssim is not None:
            self.cfg.update({'ssim': ssim})

    def dump(para_dir):
        t = time()
        cfg_file = "{}/{}.json".format(para_dir, t)
        with open() as f:
            json.dump(self.cfg, f)

    def load(self, cfg_file):
        try:
            with open(d_cfg_file, "w") as f:
                self.cfg = json.load(f)
        except FileNotFoundError:
            print("Error! Cannot find cfg for " + cfg_file)
