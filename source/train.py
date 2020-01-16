# normal
import argparse
import utils.configs as cfg

FRAME = 8
VALIDATE_AFTER = 10
TRAIN_VLIDATE_SPLIT = 0.9

def train_e2e():
    pass 

def train_denoiser():
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
    args = parser.parse_args()

    args.trainer()
