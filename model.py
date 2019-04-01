#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


def calculate_padding(kernel_size):
    return int((kernel_size-1)/2)


# kernel_size must be an odd number
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, prelu=False):
        super().__init__()
        padding = calculate_padding(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.PReLU() if prelu else nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class BatchConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, prelu=False):
        super().__init__()
        padding = calculate_padding(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.PReLU() if prelu else nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.bnorm(self.conv(x)))
        return x
