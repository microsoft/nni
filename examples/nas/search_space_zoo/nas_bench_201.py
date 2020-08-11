import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nni.nas.pytorch.utils import AverageMeterGroup
from nni.nas.pytorch.nas_bench_201 import NASBench201Cell


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
                 bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=bn_affine, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats)
        )

    def forward(self, x):
        return self.op(x)


class ResNetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, bn_affine=True,
                 bn_momentum=0.1, bn_track_running_stats=True):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, bn_affine, bn_momentum, bn_track_running_stats)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, bn_affine, bn_momentum, bn_track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, bn_affine, bn_momentum, bn_track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        return inputs + basicblock


class NASBench201Network(nn.Module):
    def __init__(self, stem_out_channels, num_modules_per_stack, bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(NASBench201Network, self).__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = 10

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=self.bn_momentum)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicBlock(C_prev, C_curr, 2, self.affine, self.momentum, self.bn_track_running_stats)
            else:
                cell = NASBench201Cell(i, C_prev, C_curr, 1, self.bn_affine, self.momentum, self.bn_track_running_stats)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=self.bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
