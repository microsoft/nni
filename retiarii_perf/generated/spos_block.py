# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import re

import torch
import torch.nn as nn


class ShuffleNetBlock(nn.Module):
    """
    When stride = 1, the block receives input with 2 * inp channels. Otherwise inp channels.
    """

    def __init__(self, inp, oup, mid_channels, ksize, stride, sequence="pdp"):
        super().__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        self.channels = inp // 2 if stride == 1 else inp
        self.inp = inp
        self.oup = oup
        self.mid_channels = mid_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = ksize // 2
        self.oup_main = oup - self.channels
        assert self.oup_main > 0

        self.branch_main = nn.Sequential(*self._decode_point_depth_conv(sequence))

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.channels, self.channels, ksize, stride, self.pad,
                          groups=self.channels, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                # pw-linear
                nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.channels, affine=False),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 2:
            x_proj, x = self.branch_proj(x), x
        else:
            x_proj, x = self._channel_shuffle(x)
        return torch.cat((x_proj, self.branch_main(x)), 1)

    def _decode_point_depth_conv(self, sequence):
        result = []
        first_depth = first_point = True
        pc = c = self.channels
        for i, token in enumerate(sequence):
            # compute output channels of this conv
            if i + 1 == len(sequence):
                assert token == "p", "Last conv must be point-wise conv."
                c = self.oup_main
            elif token == "p" and first_point:
                c = self.mid_channels
            if token == "d":
                # depth-wise conv
                assert pc == c, "Depth-wise conv must not change channels."
                result.append(nn.Conv2d(pc, c, self.ksize, self.stride if first_depth else 1, self.pad,
                                        groups=c, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                first_depth = False
            elif token == "p":
                # point-wise conv
                result.append(nn.Conv2d(pc, c, 1, 1, 0, bias=False))
                result.append(nn.BatchNorm2d(c, affine=False))
                result.append(nn.ReLU(inplace=True))
                first_point = False
            else:
                raise ValueError("Conv sequence must be d and p.")
            pc = c
        return result

    def _channel_shuffle(self, x):
        bs, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(bs * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleXceptionBlock(ShuffleNetBlock):

    def __init__(self, inp, oup, mid_channels, stride):
        super().__init__(inp, oup, mid_channels, 3, stride, "dpdpdp")


class ShuffleNetV2OneShot(nn.Module):
    block_keys = [
        'shufflenet_3x3',
        'shufflenet_5x5',
        'shufflenet_7x7',
        'xception_3x3',
    ]

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024, n_classes=1000,
                 op_flops_path="../data/op_flops_dict.pkl"):
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

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=False),
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
            nn.BatchNorm2d(last_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        from nni.nas.pytorch import mutables
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            choice_block = mutables.LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride)
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
    result = dict()
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[len("module."):]
        k = re.sub(r"^(features.\d+).(\d+)", "\\1.choices.\\2", k)
        result[k] = v
    return result


if __name__ == "__main__":
    model = ShuffleNetV2OneShot()
    print(model._parsed_flops)
