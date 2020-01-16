from time import time
import json
import os

def general(name):
    pixel = 256
    train_file = './data/train/train%s%d.mat' % (name, pixel)
    test_file = './data/test/test%s%d.mat' % (name, pixel)
    mask_file = './data/mask256.mat' % (pixel)
    para_dir = './results/parameter/Para%s%d' % (name, pixel)
    recon_dir = './results/reconstruction/Img%s%d' % (name, pixel)
    model_dir = './results/model/Model%s%d' % (name, pixel)
    if not os.path.exist(para_dir):
        os.path.makedirs(para_dir)
    if not os.path.exist(recon_dir):
        os.path.makedirs(recon_dir)        
    if not os.path.exist(model_dir):
        os.path.makedirs(model_dir)
    return train_file, test_file, mask_file, para_dir, recon_dir, model_dir

class Config(object):
    def __init__(self,u_name,d_name,u_cfg,d_cfg):
        self.u_name = u_name
        self.d_name = d_name
        self.u_cfg = u_cfg
        self.d_cfg = d_cfg

    def dump(para_dir):
        t = time() 
        u_cfg_file = "{}/{}_u.json".format(para_dir,t)
        d_cfg_file = "{}/{}_d.json".format(para_dir,t)
        with open() as f:
            json.dump(self.u_cfg,f)
        with open() as f:
            json.dump(self.d_cfg,f)

    def load_u(self,u_cfg_file):
        try:
            with open(u_cfg_file, "w") as f:
                self.u_cfg = json.load(f)
        except FileNotFoundError:
            print("Error! Cannot find cfg for " + u_cfg_file)
        
    def load_d(self,d_cfg_file):
        try:
            with open(d_cfg_file, "w") as f:
                self.d_cfg = json.load(f)
        except FileNotFoundError:
            print("Error! Cannot find cfg for " + d_cfg_file)

    def load(self,u_cfg_file,d_cfg_file):
        self.load_u(u_cfg_file)
        self.load_d(d_cfg_file)
