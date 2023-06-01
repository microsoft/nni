# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from lib.ops import (
    MBBlock,
    SeparableConv,
    SingleOperation,
    StemBlock,
    conv_bn,
)
from torch.nn import init

INIT_CH = 16


class PFLDInference(nn.Module):
    """ The subnet with the architecture of PFLD. """

    def __init__(self, lookup_table, sampled_ops, num_points=106):
        """
        Parameters
        ----------
        lookup_table : class
            to manage the candidate ops, layer information and layer perf
        sampled_ops : list of str
            the searched layer names of the subnet
        num_points : int
            the number of landmarks for prediction
        """
        super(PFLDInference, self).__init__()

        stage_names = [stage_name for stage_name in lookup_table.layer_num]
        stage_n = [lookup_table.layer_num[stage] for stage in stage_names]
        self.stem = StemBlock(init_ch=INIT_CH, bottleneck=False)

        self.block4_1 = MBBlock(INIT_CH, 32, stride=2, mid_ch=32)
        stages_0 = [
            SingleOperation(
                lookup_table.layer_configs[layer_id],
                lookup_table.lut_ops[stage_names[0]],
                sampled_ops[layer_id],
            )
            for layer_id in range(stage_n[0])
        ]

        stages_1 = [
            SingleOperation(
                lookup_table.layer_configs[layer_id],
                lookup_table.lut_ops[stage_names[1]],
                sampled_ops[layer_id],
            )
            for layer_id in range(stage_n[0], stage_n[0] + stage_n[1])
        ]

        blocks = stages_0 + stages_1
        self.blocks = nn.Sequential(*blocks)

        self.avg_pool1 = nn.Conv2d(
            INIT_CH, INIT_CH, 9, 8, 1, groups=INIT_CH, bias=False
        )
        self.avg_pool2 = nn.Conv2d(32, 32, 3, 2, 1, groups=32, bias=False)

        self.block6_1 = nn.Conv2d(96 + INIT_CH, 64, 1, 1, 0, bias=False)
        self.block6_2 = MBBlock(64, 64, res=True, se=True, mid_ch=128)
        self.block6_3 = SeparableConv(64, 128, 1)

        self.conv7 = nn.Conv2d(128, 128, 7, 1, 0, groups=128, bias=False)
        self.fc = nn.Conv2d(128, num_points * 2, 1, 1, 0, bias=True)

        # init params
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            input image

        Returns
        -------
        output: tensor
            the predicted landmarks
        output: tensor
            the intermediate features
        """
        x, y1 = self.stem(x)
        out1 = x

        x = self.block4_1(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:
                y2 = x
            elif i == 4:
                y3 = x

        y1 = self.avg_pool1(y1)
        y2 = self.avg_pool2(y2)
        multi_scale = torch.cat([y3, y2, y1], 1)

        y = self.block6_1(multi_scale)
        y = self.block6_2(y)
        y = self.block6_3(y)
        y = self.conv7(y)
        landmarks = self.fc(y)

        return landmarks, out1


class AuxiliaryNet(nn.Module):
    """ AuxiliaryNet to predict pose angles. """

    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(INIT_CH, 64, 3, 2)
        self.conv2 = conv_bn(64, 64, 3, 1)
        self.conv3 = conv_bn(64, 32, 3, 2)
        self.conv4 = conv_bn(32, 64, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            input intermediate features

        Returns
        -------
        output: tensor
            the predicted pose angles
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
