import numpy as np
import torch


class Masker():
    """Object for masking and demasking"""

    def __init__(self, frame=8, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False, mask_3d=False):
        self.frame = frame
        self.grid_size = width
        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input
        self.mask_3d = mask_3d
        if self.mask_3d:
            self.n_masks = width ** 3  # to iterate every pixel in the mask patches
        else:
            self.n_masks = width ** 2

    def mask(self, X, i):
        phase_x = i % self.grid_size
        phase_y = (i // self.grid_size) % self.grid_size
        if self.mask_3d: 
            phase_t = (i // (self.grid_size*self.grid_size)) % self.grid_size
            mask = pixel_grid_mask(X.shape, self.grid_size, phase_x, phase_y, phase_t)
        else:
            mask = pixel_grid_mask(X.shape, self.grid_size, phase_x, phase_y)
        mask = mask.to(X.device)

        mask_inv = torch.ones(mask.shape).to(X.device) - mask

        if self.mode == 'interpolate':
            masked = interpolate_mask(X, mask, mask_inv, self.frame)
        elif self.mode == 'zero':
            masked = X * mask_inv
        else:
            raise NotImplementedError

        if self.include_mask_as_input:
            net_input = torch.cat(
                (masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, X, model):
        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat(
                    (X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
            else:
                net_input = X
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(X, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).cpu()

            for i in range(self.n_masks):
                net_input, mask = self.mask(X, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).cpu()

            return acc_tensor


def pixel_grid_mask(shape, patch_size, phase_x, phase_y, phase_t=1):
    A = torch.zeros(shape)
    for i in range(shape[-3]):
        for j in range(shape[-2]):
            for k in range(shape[-1]):
                if (i % patch_size == phase_t and j % patch_size == phase_x and k % patch_size == phase_y):
                    A[..., i, j, k] = 1
    return A


def interpolate_mask(tensor, mask, mask_inv, channels):
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
    kernel = np.tile(kernel,(channels,1,1))
    kernel = kernel[np.newaxis, ...]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv
