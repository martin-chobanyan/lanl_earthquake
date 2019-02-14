#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


ROOT_FOLDER = '/home/mchobanyan/data/kaggle/lanl_earthquake/'


class EarthquakePlotViewer(object):
    """Returns plots of the earthquake time series (based on the pytorch Dataset class)"""

    def __init__(self, folder, acoustic_bounds=None, time_bounds=None):
        self.test_folder = folder
        self.file_names = os.listdir(folder)
        self.acoustic_bounds = acoustic_bounds
        self.time_bounds = time_bounds

    def __getitem__(self, item):

        earthquake_ts = pd.read_csv(os.path.join(self.test_folder, self.file_names[item]))

        fig, ax1 = plt.subplots()
        ax1.plot(list(range(len(earthquake_ts))), earthquake_ts['acoustic_data'], 'b')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Acoustic', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_title(self.file_names[item])
        if self.acoustic_bounds is not None:
            ax1.set_ylim(self.acoustic_bounds)

        if 'time_to_failure' in earthquake_ts.columns:
            ax2 = ax1.twinx()
            ax2.plot(list(range(len(earthquake_ts))), earthquake_ts['time_to_failure'], 'g')
            ax2.set_ylabel('Time to failure', color='g')
            ax2.tick_params('y', colors='g')
            if self.time_bounds is not None:
                ax2.set_ylim(self.time_bounds)

        return fig, self.file_names[item]

    def __len__(self):
        return len(self.file_names)


def save_plots(mode, acoustic_bounds=None, time_bounds=None):
    """Save plots as images

    Parameters
    ----------
    mode: str
        Specifies which plots to save, 'train' or 'test'
    acoustic_bounds: tuple
        The optional bounds for the acoustic values (default=None)
    time_bounds: tuple
        The optional bounds for the time left (default=None)
    """
    plots = EarthquakePlotViewer(os.path.join(ROOT_FOLDER, mode), acoustic_bounds, time_bounds)
    for i in range(len(plots)):
        print(f'Files complete: {i}')
        fig, filename = plots[i]
        fig.savefig(os.path.join(ROOT_FOLDER, mode + '_images', filename + '.png'), format='png')
        plt.close()


if __name__ == '__main__':
    save_plots('train', acoustic_bounds=(-5000, 5000), time_bounds=(0, 16))
