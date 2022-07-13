######################################
#         Kaihua Tang
######################################

import math
from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, is_last=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNext(nn.Module):
    def __init__(self, block, layers, groups=1, width_per_group=64, num_experts=1, reduce_dimension=False):
        self.inplanes = 64
        self.num_experts = num_experts
        super(ResNext, self).__init__()

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        if reduce_dimension:
            layer3_output_dim = 192
        else:
            layer3_output_dim = 256

        if reduce_dimension:
            layer4_output_dim = 384
        else:
            layer4_output_dim = 512
        
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks-1)))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, index=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if index is None:
            feats = []
            for ind in range(self.num_experts):
                feats.append(self._separate_part(x, ind))
            return torch.stack(feats, dim=1)
        else:
            return self._separate_part(x, index)

class ResNeXt50Model(nn.Module):
    def __init__(self, reduce_dimension=False, num_experts=1):
        super(ResNeXt50Model, self).__init__()
        self.backbone = ResNext(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, reduce_dimension=reduce_dimension, num_experts=num_experts)
    def forward(self, x, index=None):
        return self.backbone(x, index)


def create_model(m_type='resnext50', num_experts=3, reduce_dimension=True):
    # create various resnet models
    if m_type == 'resnext50':
        model = ResNeXt50Model(reduce_dimension=reduce_dimension, num_experts=num_experts)
    return model