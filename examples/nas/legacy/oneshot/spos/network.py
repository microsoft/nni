# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import re

import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii.nn.pytorch import LayerChoice
from nni.retiarii.serializer import model_wrapper

from blocks import ShuffleNetBlock, ShuffleXceptionBlock


@model_wrapper
class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024,
                 n_classes=1000, affine=False):
        super().__init__()

        assert input_size % 32 == 0
        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes
        self._affine = affine
        self._layerchoice_count = 0

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self._feature_map_size //= 2

        p_channels = first_conv_channels
        features = []
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            self._layerchoice_count += 1
            choice_block = LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride, affine=self._affine),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride, affine=self._affine)
            ], label="LayerChoice" + str(self._layerchoice_count))
            result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                else:
                    torch.nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0001)
                torch.nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)


def load_and_parse_state_dict(filepath="./data/checkpoint-150000.pth.tar"):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    result = dict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[len("module."):]
        result[k] = v
    return result
