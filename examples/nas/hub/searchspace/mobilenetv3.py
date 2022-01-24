import math
from typing import Tuple

import torch.nn as nn

from nni.retiarii import model_wrapper

from .proxylessnas import ConvBNReLU, InvertedResidual, ProxylessSpace, SeparableConv, make_divisible


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


@model_wrapper
class MobileNetV3(ProxylessSpace):
    """
    We use the following snipppet as reference.
    https://github.com/google-research/google-research/blob/20736344591f774f4b1570af64624ed1e18d2867/tunas/mobile_search_space_v3.py#L728
    """
    def __init__(self, num_labels: int = 1000,
                 base_widths: Tuple[int, ...] = (16, 16, 32, 64, 128, 256, 512, 1024),
                 width_multipliers: Tuple[float, ...] = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
                 expansion_multipliers: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                 dropout_rate: float = 0.,
                 stem_width: int = 32,
                 width_mult: float = 1.0,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.1):
        super().__init__()

        self.widths = []
        self.width_multipliers = width_multipliers

        act = h_swish

        blocks = [
            # Stem
            ConvBNReLU(
                3, self._get_width(base_widths[0]),
                nn.ValueChoice([3, 5], label='first_conv_ks'),
                stride=2, activation_layer=act
            ),
            SeparableConv(self.widths[-1], self._get_width(base_widths[0]), activation_layer=nn.ReLU),
        ]

        for stage_idx in range(1, 6):
            blocks += [
                
            ]

        # building first layer
        input_channel = make_divisible(base_widths * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = make_divisible(c * width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _get_width(self, base_width):
        new_width = nn.ValueChoice([make_divisible(base_width * mult, 8) for mult in self.width_multipliers],
                                   label=f'width_{len(self.widths)}')
        self.widths.append(new_width)
        return new_width


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
