######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models

import math

class ClassifierFC(nn.Module):
    def __init__(self, feat_dim, num_classes=1000):
        super(ClassifierFC, self).__init__()
        
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x, add_inputs=None):
        y = self.fc(x)
        return y


def create_model(feat_dim=2048, num_classes=1000):
    model = ClassifierFC(feat_dim=feat_dim, num_classes=num_classes)
    return model