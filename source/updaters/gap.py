import torch.nn as nn
import torch


class Gap(nn.Module):
    def __init__(self,phi,y,denoiser,step_size,**kwgs):
        super(Gap, self).__init__()
        self.phi = phi
        self.frame = phi.shape[0]
        self.phi_sum = self.phi.sum(0)
        self.y = y
        self.denoiser = denoiser
        self.step_size = step_size
        if 'acc' in kwgs:
            self.acc = kwgs['acc']
        else:
            self.acc = False

    def forward(self, *params):
        x, delta_y = params
        y_t = x.mul(self.phi).sum(1)
        if self.acc:
            delta_y = delta_y + (delta_y - y_t)
            residual = (delta_y-y_t).repeat(self.frame,1,1,1).permute(1,0,2,3)
            delta_x = self.step_size*residual.mul(self.phi).div(self.phi_sum+0.0001) 
        else:
            residual = (self.y-y_t).repeat(self.frame,1,1,1).permute(1,0,2,3)
            delta_x = self.step_size*residual.mul(self.phi).div(self.phi_sum+0.0001)
        x = x + delta_x
        x = self.denoiser(x)
        return x, delta_y

    def initialize(self):
        delta_y = torch.zeros_like(self.y)
        return [delta_y]

