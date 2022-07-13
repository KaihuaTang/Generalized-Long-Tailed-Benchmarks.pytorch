######################################
#         Kaihua Tang
######################################
import math
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


from .DT_COCO_LT import COCO_LT
from .DT_ColorMNIST import ColorMNIST_LT
from .DT_ImageNet_LT import ImageNet_LT

from .Sampler_ClassAware import ClassAwareSampler
from .Sampler_MultiEnv import WeightedSampler, DistributionSampler, FixSeedSampler

##################################
# return a dataloader
##################################
def get_loader(config, phase, testset, logger):
    if config['dataset']['name'] in ('MSCOCO-LT', 'MSCOCO-BL'):
        split = COCO_LT(phase=phase, 
                             data_path=config['dataset']['data_path'], 
                             anno_path=config['dataset']['anno_path'],
                             testset=testset,
                             rgb_mean=config['dataset']['rgb_mean'],
                             rgb_std=config['dataset']['rgb_std'],
                             rand_aug = config['dataset']['rand_aug'],
                             output_path=config['output_dir'], 
                             logger=logger)
    elif config['dataset']['name'] in ('ColorMNIST-LT', 'ColorMNIST-BL'):
        split = ColorMNIST_LT(phase=phase,
                              testset=testset,
                              data_path=config['dataset']['data_path'],
                              cat_ratio=config['dataset']['cat_ratio'], 
                              att_ratio=config['dataset']['att_ratio'],
                              rand_aug = config['dataset']['rand_aug'],
                              logger=logger)
    elif config['dataset']['name'] in ('ImageNet-LT', 'ImageNet-BL'):
        split = ImageNet_LT(phase=phase,
                             anno_path=config['dataset']['anno_path'],
                             testset=testset,
                             rgb_mean=config['dataset']['rgb_mean'],
                             rgb_std=config['dataset']['rgb_std'],
                             rand_aug = config['dataset']['rand_aug'],
                             output_path=config['output_dir'], 
                             logger=logger)
    else:
        logger.info('********** ERROR: unidentified dataset **********')
    
    
    

    # create data sampler
    sampler_type = config['sampler']

    # class aware sampling (re-balancing)
    if sampler_type == 'ClassAwareSampler' and phase == 'train':
        logger.info('======> Sampler Type {}'.format(sampler_type))
        sampler = ClassAwareSampler(split, num_samples_cls=4)
        loader = data.DataLoader(split, num_workers=config['training_opt']['data_workers'],
                                    batch_size=config['training_opt']['batch_size'],
                                    sampler=sampler,
                                    pin_memory=True,)
    # hard weighted sampling (don't sampling samples with weights smaller than 1.0)
    elif sampler_type == 'WeightedSampler' and phase == 'train':
        logger.info('======> Sampler Type {}, Sampler Number {}'.format(sampler_type, config['num_sampler']))
        loader = []
        num_sampler = config['num_sampler']
        batch_size =  config['training_opt']['batch_size']
        if config['batch_split']:
            batch_size = batch_size // num_sampler
        for _ in range(num_sampler):
            loader.append(data.DataLoader(split, num_workers=config['training_opt']['data_workers'],
                                    batch_size=batch_size,
                                    sampler=WeightedSampler(split),
                                    pin_memory=True,))
    # soft weighted sampling (sampling samples by the provided weights)
    elif sampler_type == 'DistributionSampler' and phase == 'train':
        logger.info('======> Sampler Type {}, Sampler Number {}'.format(sampler_type, config['num_sampler']))
        loader = []
        num_sampler = config['num_sampler']
        batch_size =  config['training_opt']['batch_size']
        if config['batch_split']:
            batch_size = batch_size // num_sampler
        for _ in range(num_sampler):
            loader.append(data.DataLoader(split, num_workers=config['training_opt']['data_workers'],
                                    batch_size=batch_size,
                                    sampler=DistributionSampler(split),
                                    pin_memory=True,))
    # Random Sampling with given seed
    elif sampler_type == 'FixSeedSampler' and phase == 'train':
        logger.info('======> Sampler Type {}, Sampler Number {}'.format(sampler_type, config['num_sampler']))
        loader = []
        num_sampler = config['num_sampler']
        batch_size =  config['training_opt']['batch_size']
        if config['batch_split']:
            batch_size = batch_size // num_sampler
        for _ in range(num_sampler):
            loader.append(data.DataLoader(split, num_workers=config['training_opt']['data_workers'],
                                    batch_size=batch_size,
                                    sampler=FixSeedSampler(split),
                                    pin_memory=True,))
    else:
        logger.info('======> Sampler Type Naive Sampling with shuffle type: {}'.format(True if phase == 'train' else False))
        loader = data.DataLoader(split, num_workers=config['training_opt']['data_workers'],
                                    batch_size=config['training_opt']['batch_size'],
                                    shuffle=True if phase == 'train' else False,
                                    pin_memory=True,)

    return loader









