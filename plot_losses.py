#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
from dtsckit.utils import read_pickle

if __name__ == '__main__':
    root_folder = '/home/mchobanyan/data/kaggle/lanl_earthquake/'

    training_losses = read_pickle(os.path.join(root_folder, 'training_losses.pkl'))
    testing_losses = read_pickle(os.path.join(root_folder, 'testing_losses.pkl'))

    plt.plot(range(len(training_losses)), training_losses, label='train')
    plt.plot(range(len(testing_losses)), testing_losses, label='test')
    plt.legend(loc='best')
    plt.show()
