# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class ShuffleNetBlock(nn.Module):
    """
    When stride = 1, the block receives input with 2 * inp channels. Otherwise inp channels.
    """

    def __init__(self, inp, oup, mid_channels, ksize, stride, sequence="pdp", affine=True):
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
        self._affine = affine
        assert self.oup_main > 0

        self.branch_main = nn.Sequential(*self._decode_point_depth_conv(sequence))

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.channels, self.channels, ksize, stride, self.pad,
                          groups=self.channels, bias=False),
                nn.BatchNorm2d(self.channels, affine=affine),
                # pw-linear
                nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.channels, affine=affine),
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
                result.append(nn.BatchNorm2d(c, affine=self._affine))
                first_depth = False
            elif token == "p":
                # point-wise conv
                result.append(nn.Conv2d(pc, c, 1, 1, 0, bias=False))
                result.append(nn.BatchNorm2d(c, affine=self._affine))
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

    def __init__(self, inp, oup, mid_channels, stride, affine=True):
        super().__init__(inp, oup, mid_channels, 3, stride, "dpdpdp", affine)
