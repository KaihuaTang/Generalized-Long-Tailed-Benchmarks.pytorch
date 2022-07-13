######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierMultiHead(nn.Module):
    def __init__(self, feat_dim, num_classes=1000):
        super(ClassifierMultiHead, self).__init__()
        
        self.fc1 = nn.Linear(feat_dim, num_classes, bias=False)
        self.fc2 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.fc3 = nn.Linear(feat_dim, feat_dim, bias=False)

    def forward(self, x, add_inputs=None):
        y1 = self.fc1(x.detach())  # prediction (re-train in stage2)
        y2 = self.fc2(x)           # contrastive head
        y3 = self.fc3(x)           # metric head
        return y1, y2, y3


def create_model(feat_dim=2048, num_classes=1000):
    model = ClassifierMultiHead(feat_dim=feat_dim, num_classes=num_classes)
    return model