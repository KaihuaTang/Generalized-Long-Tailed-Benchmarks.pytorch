######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import math

class ClassifierLDAM(nn.Module):
    def __init__(self, feat_dim, num_classes=1000):
        super(ClassifierLDAM, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes).cuda(), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
        self.weight.data.renorm_(2, 1, 1e-5)
        self.weight.data.mul_(1e5)


    def forward(self, x, add_inputs=None):
        y = torch.mm(F.normalize(x, dim=1), F.normalize(self.weight, dim=0))
        return y


def create_model(feat_dim=2048, num_classes=1000):
    model = ClassifierLDAM(feat_dim=feat_dim, num_classes=num_classes)
    return model