#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def sort_pad_pack(batch, n_features):
    """Pads a batch of sequences to the same length and returns a PackedSequnce

    Parameters
    ----------
    batch: list
        A list of the array_like sequences
    n_features: int
        The number of features per element in the sequences

    Returns
    -------
    PackedSequence, LongTensor
        A pytorch PackedSequence object and a tensor containing the
        indices to sort the sequences by length in descending order
    """
    seq_lengths = torch.LongTensor([len(seq) for seq in batch])

    # pad the batch tensor
    batch_tensor = torch.zeros((len(batch), seq_lengths.max(), n_features))
    for i, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
        batch_tensor[i, :seqlen] = torch.Tensor(seq)

    # sort the tensor by the sequence lengths, make the sequence axis first
    seq_lengths, sort_idx = seq_lengths.sort(0, descending=True)
    batch_tensor = batch_tensor[sort_idx]
    batch_tensor = batch_tensor.transpose(0, 1)

    # pack the padded input
    packed_input = pack_padded_sequence(batch_tensor, seq_lengths.numpy())
    return packed_input, sort_idx


class FullModel(nn.Module):
    """The full model used for submission

    This model is comprised of the following modules:
    - rnn: a 2-layer GRU operating on the hand-engineered features extracted from the acoustic segments
    - acoustic_conv: a 1D strided, BatchNorm CNN operating directly on the acoustic values of the sequence
    - fourier_conv: a 2D CNN operating on the spectograms of each acoustic sequence
    - final_map: a two linear layers that map the concatenated results of the previous modules to the predicted time
    """

    def __init__(self, num_seg_features):
        super().__init__()
        self.rnn = nn.GRU(num_seg_features, 64, 2)
        self.acoustic_conv = AcousticConv()
        self.fourier_conv = FourierConv()
        self.final_map = nn.Sequential(nn.Linear(272, 64), nn.LeakyReLU(), nn.Dropout(), nn.Linear(64, 1))

    def forward(self, acoustic, segments, fourier):
        x1 = self.acoustic_conv(acoustic)

        _, hidden_vector = self.rnn(segments)
        x2 = hidden_vector[-1]

        x3 = self.fourier_conv(fourier)

        x = torch.cat([x1, x2, x3], 1)
        x = self.final_map(x)
        return x


class FourierConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 7, padding=3), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, padding=2), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU())
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Sequential(nn.Linear(5325, 1), nn.LeakyReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(-1, 5325)
        x = self.fc(x)
        x = x.view(-1, 64)
        return x


class AcousticConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 8, 301, padding=150), nn.BatchNorm1d(8), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(8, 16, 301, stride=10), nn.BatchNorm1d(16), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(16, 32, 101, padding=50), nn.BatchNorm1d(32), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(32, 64, 101, stride=10), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(64, 128, 51, padding=25), nn.BatchNorm1d(128), nn.LeakyReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(128, 1, 51, stride=10), nn.LeakyReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x.view(-1, 144)
