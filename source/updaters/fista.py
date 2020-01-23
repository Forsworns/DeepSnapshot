import torch.nn as nn

class Fista(nn.Module):
    def __init__(self,phi,y,denoiser,step_size,**kwgs):
        super(Fista, self).__init__()
        self.frame = phi.shape[0]
        self.phi = phi
        self.y = y
        self.denoiser = denoiser
        self.step_size = step_size
        # self.volatile = volatile
        # self.requires_grad = requires_grad

    def forward(self, *params):
        # params: x, t
        last_x, last_t = params
        y_t = last_x.mul(self.phi).sum(1)
        x = last_x - self.step_size*(y_t-self.y).repeat(self.frame,1,1,1).permute(1,0,2,3).mul(self.phi)
        t = (1+(1+4*last_t**2)**0.5)/2
        x = x + (last_t-1)/t*(x-last_x)
        x = self.denoiser(x)
        return x, t
    
    def initialize(self):
        t = 1
        return [t]
