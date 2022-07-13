######################################
#         Kaihua Tang
######################################

import os
import json
import math
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F 
from torch.optim.optimizer import Optimizer, required

from utils.general_utils import *

from torch import Tensor
from typing import List, Optional


def create_optimizer(model, classifier, logger, config):
    training_opt = config['training_opt']
    lr = training_opt['optim_params']['lr']
    weight_decay = training_opt['optim_params']['weight_decay']

    # IMPORTANT
    # when the deadline is approaching, I suddenly found that I forgot to add momentum into my SGD optimizer.
    # therefore, I have to just accept the setting of 0 momentum, but since all the methods are replemented 
    # under the same optimizer, our conclusions and analyses still hold
    # For the follower, please remember to add momentum here.

    logger.info('=====> Create optimizer')
    all_params = []

    for _, val in model.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [{"params": [val], "lr": lr, "weight_decay": weight_decay}]
    for _, val in classifier.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [{"params": [val], "lr": lr, "weight_decay": weight_decay}]
    
    if training_opt['optimizer'] == 'Adam':
        return optim.Adam(all_params)
    elif training_opt['optimizer'] == 'SGD':
        return optim.SGD(all_params)
    else:
        logger.info('********** ERROR: unidentified optimizer **********')


def create_optimizer_stage2(model, classifier, logger, config):
    training_opt = config['training_opt']
    lr = training_opt['optim_params']['lr']
    weight_decay = training_opt['optim_params']['weight_decay']

    # IMPORTANT
    # when the deadline is approaching, I suddenly found that I forgot to add momentum into my SGD optimizer.
    # therefore, I have to just accept the setting of 0 momentum, but since all the methods are replemented 
    # under the same optimizer, our conclusions and analyses still hold
    # For the follower, please remember to add momentum here.

    logger.info('=====> Create optimizer')
    all_params = []

    # in two-stage training, the second stage should freeze the backbone
    logger.info('========= Freeze Backbone Parameters ===========')
    for _, val in model.named_parameters():
        val.requires_grad = False

    for _, val in classifier.named_parameters():
        if not val.requires_grad:
            continue
        all_params += [{"params": [val], "lr": lr, "weight_decay": weight_decay}]
    
    if training_opt['optimizer'] == 'Adam':
        return optim.Adam(all_params)
    elif training_opt['optimizer'] == 'SGD':
        return optim.SGD(all_params)
    else:
        logger.info('********** ERROR: unidentified optimizer **********')


def create_scheduler(optimizer, logger, config):
    training_opt = config['training_opt']

    logger.info('=====> Create Scheduler')
    scheduler_params = training_opt['scheduler_params']

    if training_opt['scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, training_opt['num_epochs'], eta_min=scheduler_params['endlr'])
    elif training_opt['scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(optimizer, gamma=scheduler_params['gamma'], step_size=scheduler_params['step_size'])
    elif training_opt['scheduler'] == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, gamma=scheduler_params['gamma'], milestones=scheduler_params['milestones'])
    else:
        logger.info('********** ERROR: unidentified optimizer **********')



def create_loss(logger, config, train_loader):
    training_opt = config['training_opt']

    if training_opt['loss'] == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    elif training_opt['loss'] == 'Focal':
        loss = FocalLoss(gamma=2.0)
    elif training_opt['loss'] == 'BalancedSoftmax':
        loss = BlSoftmaxLoss(train_loader)
    elif training_opt['loss'] == 'LDAM':
        loss = LDAMLoss(train_loader, total_epoch=training_opt['num_epochs'])
    elif training_opt['loss'] == 'RIDE':
        loss = RIDELoss(train_loader, additional_diversity_factor=config['algorithm_opt']['diversity_factor'])
    elif training_opt['loss'] == 'TADE':
        loss = TADELoss(train_loader, tau=config['algorithm_opt']['tau'])
    else:
        logger.info('********** ERROR: unidentified optimizer **********')
    logger.info('====== Set Loss Function to {} ======='.format(training_opt['loss']))
    return loss



class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterCosLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterCosLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def forward(self, x, labels):
        center = self.centers[labels]
        norm_c = self.l2_norm(center)
        norm_x = self.l2_norm(x)
        similarity = (norm_c * norm_x).sum(dim=-1)
        dist = 1.0 - similarity
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterTripletLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterTripletLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x, preds, labels):
        # use most likely categories as negative samples
        preds = preds.softmax(-1)
        batch_size = x.shape[0]
        idxs = torch.arange(batch_size).to(x.device)
        preds[idxs, labels] = -1
        adv_labels = preds.max(-1)[1]

        anchor = x                           # num_batch, num_dim
        positive = self.centers[labels]      # num_batch, num_dim
        negative = self.centers[adv_labels]  # num_batch, num_dim

        output = self.triplet_loss(anchor, positive, negative)
        return output



class BlSoftmaxLoss(nn.Module):
    def __init__(self, train_loader, reduction="mean"):
        super(BlSoftmaxLoss, self).__init__()
        # reduction: string. One of "none", "mean", "sum"
        label_count_array = count_dataset(train_loader)
        label_count_array = np.array(label_count_array) / np.sum(label_count_array)
        adjustments = np.log(label_count_array + 1e-12)
        adjustments = torch.from_numpy(adjustments).view(1, -1)
        self.adjustments = adjustments
        self.reduction = reduction

    def forward(self, logits, target):
        logits = logits + self.adjustments.to(logits.device)
        loss = F.cross_entropy(input=logits, target=target, reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            assert False

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()

class LDAMLoss(nn.Module):
    def __init__(self, dataloader, total_epoch, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        self.cls_num_list = count_dataset(dataloader)
        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.total_epoch = total_epoch

    def set_weight(self, epoch):
        idx = epoch // int(self.total_epoch * 0.8)
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.weight = torch.FloatTensor(per_cls_weights)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float().to(x.device)
        batch_m = torch.matmul(self.m_list.to(x.device)[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight.to(x.device))


class TADELoss(nn.Module):
    def __init__(self, dataloader, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
        cls_num_list = count_dataset(dataloader)
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss


class RIDELoss(nn.Module):
    '''
    Copy from https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/model/loss.py
    '''
    def __init__(self, dataloader=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=80, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.02):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if dataloader is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.
            cls_num_list = count_dataset(dataloader)
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def set_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        self.to(output_logits.device)
        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = logits_item
            # the following line of code is unfair (original implementation) for no diversity loss
            #ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss







