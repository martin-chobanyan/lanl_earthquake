#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from .model import Conv


class SegmentConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(1, 32, 9)
        self.conv2 = Conv(32, 64, 7)
        self.conv3 = Conv(64, 32, 5)
        self.fc1 = nn.Linear((32 * 1000), 100)
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(100, 1000)

    def encode(self, x):
        x = x.view(-1, 1, 1000)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, (32 * 1000))
        x = self.leaky_relu(self.fc1(x))
        return x

    def decode(self, x):
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
