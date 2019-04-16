#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from torch.utils.data import Dataset


# scale the data to mean=0, std=1
def scale_acoustic(data):
    scaled_data = (data - 4.5) / 10.735
    return scaled_data


class BaseDataGenerator(object):
    """Sample a random batch of a fixed-length segment from the data

    Parameters
    ----------
    data: np.ndarray
    min_index: int
        The index of the lower bound, inclusive.
    max_index: int
        The index of the upper bound, exclusive. Full data if left None (default=None).
    jump: int
        The lengths of the sampled segments.
    batch_size: int
        The number of samples in the batch (default=32).
    num_batches: int
        The number of batches to generate if iterating (default=10000).
    """
    def __init__(self, data, min_index, max_index=None, jump=150000, batch_size=32, num_batches=10000):
        if max_index is None:
            max_index = len(data) - 1
        self.data = data
        self.min_index = min_index
        self.max_index = max_index
        self.jump = min(jump, max_index - min_index)
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __call__(self):
        """Generate a batch

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First element holds the acoustic signals with shape (batch_size, jump).
            Second element holds the time left until the next earthquake from the last position of each segment.
        """
        start = np.random.randint(self.min_index, self.max_index - self.jump + 1, self.batch_size)

        acoustic_data = np.stack(self.data[i:i + self.jump, 0] for i in start)
        acoustic_data = scale_acoustic(acoustic_data)

        target_time = self.data[start + self.jump - 1, 1]
        return acoustic_data, target_time

    def __iter__(self):
        for i in range(self.num_batches):
            yield self.__call__()


class AutoEncoderDataGenerator(BaseDataGenerator):
    def __init__(self, data, min_index, max_index=None, jump=150000, batch_size=32, num_batches=10000, transforms=None):
        super().__init__(data, min_index, max_index, jump, batch_size, num_batches)
        self.transforms = transforms

    def __call__(self):
        acoustic_data, _ = super().__call__()
        if self.transforms is not None:
            return self.transforms(acoustic_data), acoustic_data
        else:
            return acoustic_data, acoustic_data


class FeatureDataGenerator(BaseDataGenerator):

    def __init__(self, data, min_index, max_index=None, nsteps=150, step=1000,
                 batch_size=32, num_batches=10000, transforms=None):
        super().__init__(data, min_index, max_index, nsteps * step, batch_size, num_batches)
        self.nsteps = nsteps
        self.step = step
        self.transforms = transforms

    def __call__(self):
        acoustic, times = super().__call__()
        acoustic_steps = acoustic.reshape(self.batch_size, self.nsteps, self.step)
        if self.transforms is not None:
            acoustic_steps = self.transforms(acoustic_steps)
        return acoustic_steps, times


class FixedSegmentDataset(Dataset):

    def __init__(self, data, n):
        super().__init__()
        self.data = data
        self.n = n

    def __getitem__(self, item):
        return self.data.loc[item:item + self.n, :].values

    def __len__(self):
        return len(self.data) - self.n + 1
