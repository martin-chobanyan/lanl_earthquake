#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from scipy.stats import kurtosis, skew
from lanl.segmenter import SpikeSegmenter
from lanl.spectral import create_spectrogram

MU = 4.5
SIGMA = 3.5


def standardize_acoustic(data, mu=MU, sigma=SIGMA):
    """Standardize the acoustic values with population mean mu and population standard deviation sigma"""
    return (data - mu) / sigma


def get_quake_indices(quake_times):
    """Get the starting indices of each earthquake segment from a pandas Series"""
    time_steps = quake_times.index[quake_times.diff() > 0.1].values
    return np.insert(time_steps, 0, 0)

def get_sample(data, seqlen=150000):
    """Sample an acoustic sequence from the data

    Parameters
    ----------
    data: array_like
        The array containing the acoustic values
    seqlen: int
        The desired length of the returned sequence (default=150000)

    Returns
    -------
    int
    """
    start = np.random.randint(0, len(data) - seqlen)
    end = start + seqlen
    sample = data[start:end]
    return sample


def extract_segment_features(segment, start, window_size):
    """Extracts hand-engineered features from acoustic segments

    The extracted features are statistics that measure the distribution of the acoustic values in the segment,
    particularly the amount of deviation and outlier values.

    Parameters
    ----------
    segment: ndarray
        A one dimensional numpy array of acoustic values
    start: int
        The starting index of this segment in the overall sequence
    window_size: int
        The window size used in the segmenter

    Returns
    -------
    list
        A list of the extracted features
    """
    x_mean = segment.mean()
    x_std = segment.std()
    x_min = segment.min()
    x_max = segment.max()
    absolute_x = np.abs(segment)
    x_abs_mean = absolute_x.mean()
    x_abs_min = absolute_x.min()
    x_abs_max = absolute_x.max()
    x_len = segment.shape[0] / window_size
    x_idx = start / window_size

    x_minmax_ratio = x_max / np.abs(x_min)
    x_minmax_diff = x_max - x_min

    x_skew = skew(segment)
    x_kurtosis = kurtosis(segment)

    features = [x_mean, x_std, x_min, x_max, x_abs_mean, x_abs_min, x_abs_max,
                x_len, x_minmax_ratio, x_minmax_diff, x_idx, x_skew, x_kurtosis]

    return features


class SegFeatureGen(object):
    """Generates training acoustic segments and builds features

    This class randomly samples a batch of acoustic sequences, segments them based on their spiked periods of acoustic
    activity, extracts statistical features from the segments, creates a spectogram from the entire track, and returns
    all of these features along with the accompany time left until the next earthquake in order to train in a supervised
    manner.

    Parameters
    ----------
    data: ndarray
        A numpy array of shape (n, 2) where n is the number of timesteps,
        the first column holds the acoustic values, and the second columns holds the time until the next earthquake
    segmenter: SpikeSegmenter
        The segmenter for getting the "excited" segments from the acoustic sequence
    batch_size: int
        The number of acoustic sequences to return per batch (default=32)
    min_index: int
        The minimum index in the data that can be included in the random sample (default=0)
    max_index: int
        The maximum index in the data that can be included in the random sample (default=None)
    seq_len: int
        The length of each acoustic sequence in the batch (default=150000)
    """

    def __init__(self, data, segmenter, batch_size=32, min_index=0, max_index=None, seq_len=150000, num_features=13):
        if max_index is None:
            max_index = len(data) - 1
        self.data = data
        self.segmenter = segmenter
        self.window_size = segmenter.window_size
        self.mu = segmenter.mu
        self.sigma = segmenter.sigma
        self.min_index = min_index
        self.max_index = max_index
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_features = num_features

    def standardize(self, acoustic):
        return standardize_acoustic(acoustic, self.mu, self.sigma)

    def __call__(self):
        rows = np.random.randint(self.min_index + self.seq_len, self.max_index, self.batch_size)
        acoustic = np.zeros((self.batch_size, 1, self.seq_len))
        segment_features = []
        spectrograms = []
        targets = []

        for idx, row in enumerate(rows):
            acoustic_seq = self.data[(row - self.seq_len):row, 0]
            cutoffs = self.segmenter(acoustic_seq)

            feature_seq = []
            acoustic_seq = self.standardize(acoustic_seq)
            for i, (s, e) in enumerate(cutoffs):
                feature_seq.append(extract_segment_features(acoustic_seq[s:e], s, self.window_size))
            segment_features.append(np.stack(feature_seq))

            _, _, spec = create_spectrogram(acoustic_seq)
            spectrograms.append(np.expand_dims(spec, 0))

            time_to_failure = self.data[row - 1, 1]
            targets.append(time_to_failure)

            acoustic[idx][0] = acoustic_seq

        acoustic = torch.Tensor(acoustic)
        spectrograms = torch.Tensor(spectrograms)
        targets = torch.Tensor(targets)

        return acoustic, segment_features, spectrograms, targets

    def __iter__(self):
        while True:
            yield self.__call__()


class TestSegFeatureGen(object):
    """This class takes an existing acoustic sequence and builds the same features as in SegFeatureGen

    The test files are already split into 150000 length acoustic sequences, without the time column.
    This is why a separate data generator class is needed for these segments.
    """
    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.mu = segmenter.mu
        self.sigma = segmenter.sigma
        self.window_size = segmenter.window_size

    def __call__(self, acoustic):
        segment_features = []
        spectrograms = []

        cutoffs = self.segmenter(acoustic)

        feature_seq = []
        acoustic = standardize_acoustic(acoustic)
        for i, (s, e) in enumerate(cutoffs):
            feature_seq.append(extract_segment_features(acoustic[s:e], s, self.window_size))
        segment_features.append(np.stack(feature_seq))

        _, _, spec = create_spectrogram(acoustic)
        spectrograms.append(np.expand_dims(spec, 0))

        acoustic = torch.Tensor(acoustic.reshape((1, 1, 150000)))
        spectrograms = torch.Tensor(spectrograms)

        return acoustic, segment_features, spectrograms
