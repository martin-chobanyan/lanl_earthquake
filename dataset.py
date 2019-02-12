#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


class TestSegmentPlotViewer(object):

    def __init__(self, test_folder):
        self.test_folder = test_folder
        self.file_names = os.listdir(test_folder)

    def __getitem__(self, item):
        fig, ax = plt.subplots()
        df = pd.read_csv(os.path.join(self.test_folder, self.file_names[item]))
        acoustic_data = df['acoustic_data'].values
        ax.plot(list(range(len(acoustic_data))), acoustic_data)
        ax.set_title(f'Acoustic data for test segment {self.file_names[item]}')
        return fig
