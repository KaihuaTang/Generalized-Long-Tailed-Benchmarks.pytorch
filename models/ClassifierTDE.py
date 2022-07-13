######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierTDE(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125):
        super(ClassifierTDE, self).__init__()

        # classifier weights
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)

        # parameters
        self.scale = tau / num_head   # 16.0 / num_head
        self.norm_scale = gamma
        self.alpha = alpha
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, add_inputs=None):            
        normed_x = self.multi_head_call(self.l2_norm, x)
        normed_w = self.multi_head_call(self.causal_norm, self.weight, weight=self.norm_scale)
        y = torch.mm(normed_x * self.scale, normed_w.t())

        # apply TDE during inference
        if (not self.training):
            self.embed = add_inputs['embed']
            normed_c = self.multi_head_call(self.l2_norm, self.embed)
            x_list = torch.split(normed_x, self.head_dim, dim=1)
            c_list = torch.split(normed_c, self.head_dim, dim=1)
            w_list = torch.split(normed_w, self.head_dim, dim=1)
            output = []

            for nx, nc, nw in zip(x_list, c_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, nc)
                y0 = torch.mm((nx -  cos_val * self.alpha * nc) * self.scale, nw.t())
                output.append(y0)
            y = sum(output)
        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, weight=None):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x
    
    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

def create_model(feat_dim=2048, num_classes=1000, num_head=2, tau=16.0, alpha=3.0, gamma=0.03125):
    model = ClassifierTDE(feat_dim=feat_dim, num_classes=num_classes, num_head=num_head, tau=tau, alpha=alpha, gamma=gamma)
    return model