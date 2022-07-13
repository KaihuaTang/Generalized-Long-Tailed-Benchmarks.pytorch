######################################
#         Kaihua Tang
######################################

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(m_type='resnet101'):
    # create various resnet models
    if m_type == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif m_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif m_type == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif m_type == 'resnext50':
        model = models.resnext50_32x4d(pretrained=False)
    elif m_type == 'resnext101':
        model = models.resnext101_32x8d(pretrained=False)
    else:
        raise ValueError('Wrong Model Type')
    model.fc = nn.ReLU()
    return model