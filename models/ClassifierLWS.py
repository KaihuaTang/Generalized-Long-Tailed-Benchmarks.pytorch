######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierLWS(nn.Module):
    def __init__(self, feat_dim, num_classes=1000):
        super(ClassifierLWS, self).__init__()

        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

        self.scales = nn.Parameter(torch.ones(num_classes))
        for _, param in self.fc.named_parameters():
            param.requires_grad = False
        
    def forward(self, x, add_inputs=None):
        y = self.fc(x)
        y *= self.scales
        return y


def create_model(feat_dim=2048, num_classes=1000):
    model = ClassifierLWS(feat_dim=feat_dim, num_classes=num_classes)
    return model