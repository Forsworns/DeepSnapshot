# normal
import argparse
import utils.configs as cfg
from time import time

FRAME = 8

def test_e2e():
    pass 

def test_iterative():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer Parameters", 
        prog="python ./train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--e2e', dest='tester', const=test_e2e, default=test_iterative, action='store_const', help="train a denoiser or reconstruction model")
    parser.add_argument('--name', default='Kobe')
    parser.add_argument('--model_file',default=None)
    args = parser.parse_args()

    if args.model_file is None:
        raise Exception("Please input the model file")
    
    args.tester()
