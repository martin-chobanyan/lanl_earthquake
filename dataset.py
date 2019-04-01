#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch.utils.data import Dataset


# scale the data to mean=0, std=1
def scale_acoustic(data):
    scaled_data = (data - 4.5) / 10.735
    return scaled_data


class BaseDataGenerator(object):
    def __init__(self, data, min_index, max_index=None,
                 nsteps=150, step_length=1000,
                 batch_size=32, num_batches=10000):
        if max_index is None:
            max_index = len(data) - 1
        self.data = data
        self.min_index = min_index
        self.max_index = max_index
        self.jump = min(nsteps * step_length, max_index - min_index)
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __call__(self):
        start = np.random.randint(self.min_index, self.max_index - self.jump, self.batch_size)

        acoustic_data = np.stack(self.data[i:i + self.jump, 0] for i in start)
        acoustic_data = scale_acoustic(acoustic_data)

        target_time = self.data[start + self.jump - 1, 1]
        return torch.from_numpy(acoustic_data), torch.from_numpy(target_time)

    def __iter__(self):
        for i in range(self.num_batches):
            yield self.__call__()


class AutoEncoderDataGenerator(BaseDataGenerator):
    def __init__(self, data, min_index, max_index=None,
                 nsteps=150, step_length=1000,
                 batch_size=32, num_batches=10000, transforms=None):
        super().__init__(data, min_index, max_index, nsteps, step_length, batch_size, num_batches)
        self.transforms = transforms

    def __call__(self):
        acoustic_data, _ = super().__call__()
        if self.transforms is not None:
            return self.transforms(acoustic_data), acoustic_data
        else:
            return acoustic_data, acoustic_data


class FixedSegmentDataset(Dataset):
    
    def __init__(self, data, n):
        super().__init__()
        self.data = data
        self.n = n
        
    def __getitem__(self, item):
        return self.data.loc[item:item+self.n, :].values
    
    def __len__(self):
        return len(self.data)-self.n+1
