#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


class EarthquakePlotViewer(object):

    def __init__(self, folder, acoustic_lim=None):
        self.test_folder = folder
        self.file_names = os.listdir(folder)
        self.acoustic_lim = acoustic_lim

    def __getitem__(self, item):

        earthquake_ts = pd.read_csv(os.path.join(self.test_folder, self.file_names[item]))

        fig, ax1 = plt.subplots()
        ax1.plot(list(range(len(earthquake_ts))), earthquake_ts['acoustic_data'], 'b')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Acoustic', color='b')
        ax1.tick_params('y', colors='b')
        ax1.set_title(self.file_names[item])
        if self.acoustic_lim is not None:
            ax1.set_ylim(self.acoustic_lim)

        if 'time_to_failure' in earthquake_ts.columns:
            ax2 = ax1.twinx()
            ax2.plot(list(range(len(earthquake_ts))), earthquake_ts['time_to_failure'], 'g')
            ax2.set_ylabel('Time to failure', color='g')
            ax2.tick_params('y', colors='g')

        return fig, self.file_names[item]
