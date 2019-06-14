#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

BLUE = '#2D2FDE'
ORANGE = '#F79E2D'


class SpikeSegmenter(object):
    """Splits a signal into segments that are "excited" / spiked

    This class takes a signal and creates regular sized bins. It then labels each bin as spiked or not, where spiked
    means that the either the sample mean has deviated from the population mean by more than two standard deviations
    or the sample standard deviation exceeds the population standard deviation. Adjacent bins with the same label are
    then merged together.

    Parameters
    ----------
    mu: float
        The population mean
    sigma: float
        The population standard deviation
    window_size: int
        The size of the bins
    only_spikes: bool
        If true, returns only the cutoffs for the excited segments (default=False)

    Returns
    -------
    list[tuple[int, int]]
        A list of tuples containing the cutoffs for each segment,
        where the end index for a tuple is equal to the start index of the next tuple.
    """
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


def plot_segments(axis, x, cutoffs, rand, mu, sigma, alpha):
    """Plot the segments of a signal on a given matplotlib axis

    Parameters
    ----------
    axis: plt.axes
        The axis on which to plot on
    x: array_like
        The raw acoustic signal
    cutoffs: list[tuple[int, int]]
        The start and end indices of each segment to plot.
    rand: bool
        If True then each segment is plotted with the default, random colors.
        Otherwise, excited segments are in blue, normal segments in orange.
    mu: float
        The population mean for the signal. Will be drawn as a horizontal line.
    sigma: float
        The population standard deviation. The up/down margin will be drawn as mu +- 2*sigma
    alpha: float
        The alpha level (transparency) of the plot
    """
    for (start, end) in cutoffs:
        segment = x[start:end]
        if rand:
            axis.plot(np.arange(start, end), segment)
        else:
            color = BLUE if segment.std() > sigma else ORANGE
            axis.plot(np.arange(start, end), segment, c=color)

    axis.axhline(mu - 2*sigma, c='black', alpha=alpha)
    axis.axhline(mu + 2*sigma, c='black', alpha=alpha)
    axis.axhline(mu, c='black')


# this is a measure of variation as another potential approach to segmentation
def cumsum_of_squares(x):
    """Given a sequence x, calculate D_k = (C_k/C_T) - (k/T) where C_k = cumsum of x^2"""
    n = len(x)
    c_k = np.cumsum(x ** 2)
    c_t = c_k[-1]
    d_k = (c_k / c_t) - (np.arange(1, n + 1)) / n
    return d_k
