import torch.nn as nn
import torch


class Gap(nn.Module):
    def __init__(self,denoiser,step_size):
        super(Gap, self).__init__()
        self.denoiser = denoiser
        self.step_size = step_size

    def forward(self, *params):
        x, y, phi, delta_y = params
        y_t = x.mul(phi).sum(1)
        frame = phi.shape[0]
        phi_sum = phi.sum(0)
        # accelerated version
        delta_y = delta_y + (delta_y - y_t)
        residual = (delta_y-y_t).repeat(frame,1,1,1).permute(1,0,2,3)
        delta_x = self.step_size*residual.mul(phi.detach()).div(phi_sum+0.0001) 
        # normal version
        # residual = (y-y_t).repeat(frame,1,1,1).permute(1,0,2,3)
        # delta_x = self.step_size*residual.mul(phi.detach()).div(phi_sum+0.0001)
        x = x + delta_x
        x = self.denoiser(x)
        return x, y, phi, delta_y

    def initialize(self,phi):
        delta_y = torch.zeros_like(phi)
        return [delta_y]

