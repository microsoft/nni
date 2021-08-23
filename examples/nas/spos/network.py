# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import re

import torch
import torch.nn as nn
from nni.nas.pytorch import mutables

from blocks import ShuffleNetBlock, ShuffleXceptionBlock


class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024, n_classes=1000,
                 op_flops_path="./data/op_flops_dict.pkl", affine=False):
        super().__init__()

        assert input_size % 32 == 0
        with open(os.path.join(os.path.dirname(__file__), op_flops_path), "rb") as fp:
            self._op_flops_dict = pickle.load(fp)

        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]
        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes
        self._affine = affine

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
            choice_block = mutables.LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride, affine=self._affine),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride, affine=self._affine),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride, affine=self._affine)
            ])
            result.append(choice_block)

            # find the corresponding flops
            flop_key = (inp, oup, mid_channels, self._feature_map_size, self._feature_map_size, stride)
            self._parsed_flops[choice_block.key] = [
                self._op_flops_dict["{}_stride_{}".format(k, stride)][flop_key] for k in self.block_keys
            ]
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

    def get_candidate_flops(self, candidate):
        conv1_flops = self._op_flops_dict["conv1"][(3, self._first_conv_channels,
                                                    self._input_size, self._input_size, 2)]
        # Should use `last_conv_channels` here, but megvii insists that it's `n_classes`. Keeping it.
        # https://github.com/megvii-model/SinglePathOneShot/blob/36eed6cf083497ffa9cfe7b8da25bb0b6ba5a452/src/Supernet/flops.py#L313
        rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes,
                                                            self._feature_map_size, self._feature_map_size, 1)]
        total_flops = conv1_flops + rest_flops
        for k, m in candidate.items():
            parsed_flops_dict = self._parsed_flops[k]
            if isinstance(m, dict):  # to be compatible with classical nas format
                total_flops += parsed_flops_dict[m["_idx"]]
            else:
                total_flops += parsed_flops_dict[torch.max(m, 0)[1]]
        return total_flops

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
