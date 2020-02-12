import torch.nn as nn

from denoisers.modules import NLResGroup, ResGroup, default_conv


# RNAN
class RnaNet(nn.Module):
    def __init__(self, conv=default_conv):
        super(RnaNet, self).__init__()
        n_resgroup = 10
        n_resblock = 16
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 1
        res_scale = 1
        n_colors = 8
        act = nn.ReLU(True)
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]
        # define body module
        modules_body_nl_low = [
            NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)]
        modules_body = [
            ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        modules_tail = [
            conv(n_feats, n_colors, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        feats_shallow = self.head(x)
        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)
        res_main = self.tail(res)
        res_clean = x + res_main
        return res_clean

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(
                    'missing keys in state_dict: "{}"'.format(missing))
