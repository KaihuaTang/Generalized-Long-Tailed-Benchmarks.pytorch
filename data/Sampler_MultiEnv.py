import random
import torch
import numpy as np
from torch.utils.data.sampler import Sampler


class WeightedSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float() # init weight


    def __iter__(self):
        selected_inds = []
        # MAKE SURE self.weight.sum() == self.num_samples
        while((self.weight >= 1.0).sum().item() > 0):
            inds = self.indexes[self.weight >= 1.0].tolist()
            selected_inds = selected_inds + inds
            self.weight = self.weight - 1.0
        selected_inds = torch.LongTensor(selected_inds)
        # shuffle
        current_size = selected_inds.shape[0]
        selected_inds = selected_inds[torch.randperm(current_size)]
        expand = torch.randperm(self.num_samples) % current_size
        indices = selected_inds[expand].tolist()

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()


class DistributionSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float() # init weight


    def __iter__(self):
        self.prob = self.weight / self.weight.sum()

        indices = torch.multinomial(self.prob, self.num_samples, replacement=True).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()


class FixSeedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, epoch):
        self.epoch = epoch

