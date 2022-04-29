# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Tuple, Optional, Callable, cast

import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

from .proxylessnas import ConvBNReLU, InvertedResidual, SeparableConv, make_divisible, reset_parameters


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


SE_DEFAULT_GATING_ACTIVATION = h_sigmoid
"""This can also be ``nn.Sigmoid``, according to
`tensorflow implementation <https://github.com/google-research/google-research/blob/20736344/tunas/mobile_search_space_v3.py#L90>`__."""


class SELayer(nn.Module):
    """Squeeze-and-excite layer."""

    def __init__(self,
                 channels: int,
                 reduction: int = 4,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if activation_layer is None:
            activation_layer = SE_DEFAULT_GATING_ACTIVATION
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, make_divisible(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channels // reduction, 8), channels),
            activation_layer()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def _se_or_skip(hidden_ch: int, label: str) -> nn.LayerChoice:
    return nn.LayerChoice({
        'identity': nn.Identity(),
        'se': SELayer(hidden_ch)
    }, label=label)


@model_wrapper
class MobileNetV3Space(nn.Module):
    """
    MobileNetV3Space implements the largest search space in `TuNAS <https://arxiv.org/abs/2008.06120>`__.

    The search dimensions include widths, expand ratios, kernel sizes, SE ratio.
    Some of them can be turned off via arguments to narrow down the search space.

    Different from ProxylessNAS search space, this space is implemented with :class:`nn.ValueChoice`.

    We use the following snipppet as reference.
    https://github.com/google-research/google-research/blob/20736344591f774f4b1570af64624ed1e18d2867/tunas/mobile_search_space_v3.py#L728
    """

    def __init__(self, num_labels: int = 1000,
                 base_widths: Tuple[int, ...] = (16, 16, 32, 64, 128, 256, 512, 1024),
                 width_multipliers: Tuple[float, ...] = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
                 expand_ratios: Tuple[float, ...] = (1., 2., 3., 4., 5., 6.),
                 dropout_rate: float = 0.2,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.1,
                 se_before_activation: bool = False):
        super().__init__()

        assert len(base_widths) == 8
        assert len(width_multipliers) == 7

        self.widths = cast(nn.ChoiceOf[int], [
            nn.ValueChoice([make_divisible(base_width * mult, 8) for mult in width_multipliers], label=f's{i}_width')
            for i, base_width in enumerate(base_widths)
        ])
        self.expand_ratios = expand_ratios

        # See the debate here: https://github.com/d-li14/mobilenetv3.pytorch/issues/18
        self.se_before_activation = se_before_activation

        blocks = [
            # Stem
            ConvBNReLU(
                3, self.widths[0],
                nn.ValueChoice([3, 5], label=f's0_i0_ks'),
                stride=2, activation_layer=h_swish
            ),
            SeparableConv(self.widths[0], self.widths[0], activation_layer=nn.ReLU),
        ]

        blocks += [
            # Body
            self._make_stage(1, self.widths[0], self.widths[1], False, 2, nn.ReLU),
            self._make_stage(2, self.widths[1], self.widths[2], True, 2, nn.ReLU),
            self._make_stage(3, self.widths[2], self.widths[3], False, 2, h_swish),
            self._make_stage(4, self.widths[3], self.widths[4], True, 1, h_swish),
            self._make_stage(5, self.widths[4], self.widths[5], True, 2, h_swish),
        ]

        # Head
        blocks += [
            ConvBNReLU(self.widths[5], self.widths[6], 1, 1, activation_layer=h_swish),
            nn.AdaptiveAvgPool2d(1),

            ConvBNReLU(self.widths[6], self.widths[7], 1, 1, norm_layer=nn.Identity, activation_layer=h_swish),
        ]

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(cast(int, self.widths[7]), num_labels),
        )

        reset_parameters(self, bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_stage(self, stage_idx, inp, oup, se, stride, act):
        def layer_builder(idx):
            exp = nn.ValueChoice(list(self.expand_ratios), label=f's{stage_idx}_i{idx}_exp')
            ks = nn.ValueChoice([3, 5, 7], label=f's{stage_idx}_i{idx}_ks')
            # if SE is true, assign a layer choice to SE
            se_block = partial(_se_or_skip, label=f's{stage_idx}_i{idx}_se') if se else None
            return InvertedResidual(
                inp if idx == 0 else oup,
                oup, exp, ks,
                stride=stride if idx == 0 else 1,  # only the first layer in each stage can have stride > 1
                squeeze_and_excite=se_block,
                activation_layer=act,
                se_before_activation=self.se_before_activation
            )

        # mutable depth
        return nn.Repeat(layer_builder, depth=(1, 4), label=f's{stage_idx}_depth')
