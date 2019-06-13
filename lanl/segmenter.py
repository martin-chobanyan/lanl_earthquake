#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

BLUE = '#2D2FDE'
ORANGE = '#F79E2D'


class SpikeSegmenter(object):
    def __init__(self, mu, sigma, window_size, only_spikes=False):
        self.mu = mu
        self.sigma = sigma
        self.window_size = window_size
        self.only_spikes = only_spikes

    def __call__(self, x):
        x_means = np.mean(x.reshape((-1, self.window_size)), axis=1)
        x_stdevs = np.std(x.reshape((-1, self.window_size)), axis=1)
        labels = (x_means > self.mu + 2 * self.sigma) | (x_means < self.mu - 2 * self.sigma) | (x_stdevs > self.sigma)

        cutoffs = [0]
        prev_status = labels[0]
        spike_labels = []  # this corresponds to the labels after segments are merged
        for i, is_spike in enumerate(labels[1:]):
            if is_spike != prev_status:
                cutoffs.append(i + 1)
                spike_labels.append(prev_status)
                prev_status = is_spike
        cutoffs.append(len(labels))
        spike_labels.append(prev_status)

        # make each segment have a clear start, end index tuple + scale them to original series
        scaled_cutoffs = [(self.window_size * i, self.window_size * j) for i, j in zip(cutoffs[:-1], cutoffs[1:])]

        if self.only_spikes:
            return [seg_cutoff for seg_cutoff, is_spike in zip(scaled_cutoffs, spike_labels) if is_spike]

        return scaled_cutoffs


def plot_segments(axis, x, cutoffs, rand, mu, margin, alpha):
    for (start, end) in cutoffs:
        segment = x[start:end]
        if rand:
            axis.plot(np.arange(start, end), segment)
        else:
            color = BLUE if segment.std() > margin else ORANGE
            axis.plot(np.arange(start, end), segment, c=color)

    axis.axhline(mu - margin, c='black', alpha=alpha)
    axis.axhline(mu + margin, c='black', alpha=alpha)
    axis.axhline(mu, c='black')

