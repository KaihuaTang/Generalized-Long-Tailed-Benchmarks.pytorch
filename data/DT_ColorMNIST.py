######################################
#         Kaihua Tang
######################################


from PIL import Image, ImageDraw
from io import BytesIO
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from randaugment import RandAugment


class ColorMNIST_LT(torchvision.datasets.MNIST):
    def __init__(self, phase, testset, data_path, logger, cat_ratio=1.0, att_ratio=0.1, rand_aug=False):
        super(ColorMNIST_LT, self).__init__(root=data_path, train=(phase == 'train'), download=True)
        # mnist dataset contains self.data, self.targets
        self.dig2label = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}
        self.dig2attri = {}
        self.colors = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}
        self.logger = logger
        self.phase = phase
        
        # ColorMNIST should not use rand augmentation, as its patterns are too simple 
        # and its color confounder could be over-written by augmentation
        assert rand_aug == False

        valid_phase = ['train', 'val', 'test']
        assert phase in valid_phase
        if phase == 'train':
            full_phase = 'train'
            # generate long-tailed data
            self.cat_ratio = cat_ratio
            self.att_ratio = att_ratio
            self.generate_lt_label(cat_ratio)
        elif phase == 'test':
            full_phase = testset
            # generate long-tailed data
            if full_phase == 'test_iid':
                self.cat_ratio = cat_ratio
                self.att_ratio = att_ratio
                self.generate_lt_label(cat_ratio)
            elif full_phase == 'test_half_bl':
                self.cat_ratio = 1.0
                self.att_ratio = att_ratio
                self.generate_lt_label(1.0)
            elif full_phase == 'test_bl':
                self.cat_ratio = 1.0
                self.att_ratio = 1.0
                self.generate_lt_label(1.0)
        else:
            full_phase = phase
            self.cat_ratio = cat_ratio
            self.att_ratio = att_ratio
            self.generate_lt_label(cat_ratio)
        logger.info('====== The Current Split is : {}'.format(full_phase))        
        
        
        
    def generate_lt_label(self, ratio=1.0):
        self.label2list = {i:[] for i in range(3)}
        for img, dig in zip(self.data, self.targets):
            label = self.dig2label[int(dig)]
            self.label2list[label].append(img)
        if ratio == 1.0:
            balance_size = min([len(val) for key, val in self.label2list.items()])
            for key, val in self.label2list.items():
                self.label2list[key] = val[:balance_size]
        elif ratio < 1.0:
            current_size = len(self.label2list[0])
            for key, val in self.label2list.items():
                max_size = len(val)
                self.label2list[key] = val[:min(max_size, current_size)]
                current_size = int(current_size * ratio)
        else:
            raise ValueError('Wrong Ratio in ColorMNIST')
        
        self.labels = []
        self.imgs = []
        for key, val in self.label2list.items():
            for item in val:
                self.labels.append(key)
                self.imgs.append(item)
            self.logger.info('Generate ColorMNIST: label {} has {} images.'.format(key, len(val)))
        
                
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        img = self.imgs[index].unsqueeze(0).repeat(3,1,1)
        label = self.labels[index]
        
        # generate tail colors
        if random.random() < self.att_ratio:
            att_label = random.randint(0,2)
        else:
            att_label = label
        color = self.colors[att_label]
            
        # assign attribute
        img = self.to_color(img, color)

        if self.phase != 'train':
            # attribute
            attribute = 1 - int(att_label == label)
            return img, label, label, attribute, index
        else:
            return img, label, label, index
    
    def to_color(self, img, rgb=[1,0,0]):
        return (img * torch.FloatTensor(rgb).unsqueeze(-1).unsqueeze(-1)).float()