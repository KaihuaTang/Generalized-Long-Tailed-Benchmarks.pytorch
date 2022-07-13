######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierCOS(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, num_head=2, tau=16.0):
        super(ClassifierCOS, self).__init__()

        # classifier weights
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.reset_parameters(self.weight)

        # parameters
        self.scale = tau / num_head   # 16.0 / num_head
        self.num_head = num_head
        self.head_dim = feat_dim // num_head

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x, add_inputs=None):
        normed_x = self.multi_head_call(self.l2_norm, x)
        normed_w = self.multi_head_call(self.l2_norm, self.weight)
        y = torch.mm(normed_x * self.scale, normed_w.t())
        return y

    def multi_head_call(self, func, x):
        assert len(x.shape) == 2
        x_list = torch.split(x, self.head_dim, dim=1)
        y_list = [func(item) for item in x_list]
        assert len(x_list) == self.num_head
        assert len(y_list) == self.num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

def create_model(feat_dim=2048, num_classes=1000, num_head=2, tau=16.0):
    model = ClassifierCOS(feat_dim=feat_dim, num_classes=num_classes, num_head=num_head, tau=tau)
    return model