#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import SGD
from visdom import Visdom
import matplotlib.pyplot as plt
from dtsckit.model import train_epoch, validate_epoch, checkpoint
from .conv_ae import SegmentConvAutoEncoder
from .dataset import AutoEncoderDataGenerator
from .split_training_file import get_quake_indices


def load_data():
    data = pd.read_pickle('/home/mchobanyan/data/kaggle/lanl_earthquake/train.pkl')
    quake_starts = get_quake_indices(data['time'])
    return data, quake_starts


def train_autoencoder(folder, device, model_idx):
    vis = Visdom()
    batch_size = 32
    num_batches = 10000
    data, quake_starts = load_data()
    train_loader = AutoEncoderDataGenerator(data.values, 0, quake_starts[13]-1, nsteps=1, batch_size=batch_size, num_batches=num_batches)
    test_loader = AutoEncoderDataGenerator(data.values, quake_starts[13], nsteps=1, batch_size=batch_size, num_batches=num_batches)

    model = SegmentConvAutoEncoder().train().to(device)
    criterion = MSELoss()
    # optimizer = optim.Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.5)

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    output_folder = os.path.join(folder, model_idx)
    for batch in range(500):
        train_loss = train_epoch(batch, model, iter(train_loader), criterion, optimizer, device, print_rate=1000)
        train_losses.append(train_loss)
        if batch % 3 == 0:
            test_loss = validate_epoch(batch, model, iter(train_loader), criterion, device, print_rate=1000)
            test_losses.append(test_loss)
            if test_loss < best_loss:
                checkpoint(model, os.path.join(output_folder, f'model_{batch}'))
                best_loss = test_loss
                print('----------------------')
                print(f'Checkpoint: New minimum validation loss: {best_loss}')
                print('----------------------')

    # visualize the results
    model = model.eval()
    with torch.no_grad():
        x = test_loader()[0][0]
        x = x.cuda().view(1, 1, -1)
        conv_filter = list(model.modules())[1]
        feature_maps = conv_filter(x)

        print(feature_maps.shape)
        x = x[0][0]
        plt.plot(np.arange(len(x)), x.cpu().squeeze().numpy())
        plt.show()

        for feature in feature_maps[0]:
            plt.plot(np.arange(len(feature)), feature.cpu().numpy())
            vis.matplot(plt, win='test')
            plt.clf()
            time.sleep(2)


if __name__ == '__main__':

    NUM_ROWS = 629145480
    root_folder = '/home/mchobanyan/data/kaggle/lanl_earthquake/'
    test_folder = os.path.join(root_folder, 'test')
    train_file = os.path.join(root_folder, 'train.csv')
    model_folder = os.path.join(root_folder, 'models')
    autoencoder_folder = os.path.join(model_folder, 'autoencoders')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_autoencoder(autoencoder_folder, device, 'model0')
