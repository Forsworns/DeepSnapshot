import torch
import scipy.io as sio
import h5py
from torch.utils.data import Dataset
import os
import numpy as np

# n x t x w x d

def measure(img, phi):
    return img.mul(phi).sum(0)

class SnapshotDataset(Dataset):
    def __init__(self, phi, labels, mode='train'):
        self.phi = phi
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.labels[index]
        return img, measure(img, self.phi)


def add_noise(img,sigma):
    return img + torch.randn(img.size())*torch.rand(sigma)


class NoisyDataset(Dataset):
    def __init__(self, labels, sigma=1, mode='train'):
        self.labels = labels
        self.sigma = sigma
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.labels[index]
        return img, add_noise(img,self.sigma)

# Load training data

def load_train_data(train_file, mask_file, mat73=False):
    if mat73 == True:                                                # if .mat file is too big, use h5py to load
        train_data = h5py.File(train_file)
        train_label = np.transpose(train_data['labels'], [3, 2, 1, 0])
    else:
        train_data = sio.loadmat(train_file)
        train_label = train_data['labels']                             # labels

    mask_data = sio.loadmat(mask_file)
    phi = mask_data['phi']                                            # mask

    del train_data, mask_data
    return torch.from_numpy(train_label).permute(0,3,1,2), torch.from_numpy(phi).permute(2,0,1)

# Load testing data


def load_test_data(test_file, mask_file, mat73=False):
    if mat73 == True:
        test_data = h5py.File(test_file)
        test_label = np.transpose(test_data['labels'], [3, 2, 1, 0])
    else:
        test_data = sio.loadmat(test_file)
        test_label = test_data['labels']

    mask_data = sio.loadmat(mask_file)
    phi = mask_data['phi']

    del test_data, mask_data
    return torch.from_numpy(test_label).permute(0,3,1,2), torch.from_numpy(phi).permute(2,0,1)
