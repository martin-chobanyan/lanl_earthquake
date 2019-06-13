#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.signal import spectrogram
import matplotlib.pyplot as plt


def create_spectrogram(sig, window_size=1200):
    """
    This function creates a spectogram using scipy and returns only the first quarter of the frequencies.
    This is solely because of fine-tuning (the acoustic sequences never achieved higher frequencies).

    Parameters
    ----------
    sig: ndarray
        The target array containing the acoustic values
    window_size: int
        The number of points to include in each bin in the spectrogram (default=1200)

    Returns
    -------
    A tuple of the frequences, times, and the cutoff spectogram as numpy arrays.
    """
    freq, times, spec = spectrogram(sig, fs=4000000, nperseg=window_size)
    n = int(len(freq) / 4)
    freq = freq[:n]
    spec = spec[:n, :]
    return freq, times, spec


def plot_fourier(spec, freq, times):
    plt.pcolormesh(times, freq, spec, cmap=plt.cm.seismic)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    plt.show()


