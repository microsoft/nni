# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Tuple, Optional, Callable, cast

import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

from .proxylessnas import ConvBNReLU, InvertedResidual, SeparableConv, make_divisible, reset_parameters


class SqueezeExcite(nn.Module):
    """Squeeze-and-excite layer.
    
    We can't use the op from ``torchvision.ops`` because it's not (yet) properly wrapped,
    and ValueChoice couldn't be processed.

    Reference:

    - https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L26
    - https://github.com/d-li14/mobilenetv3.pytorch/blob/3e6938cedcbbc5ee5bc50780ea18e644702d85fc/mobilenetv3.py#L53
    """

    def __init__(self,
                 channels: int,
                 reduction_ratio: float = 0.25,
                 gate_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        rd_channels = make_divisible(channels * reduction_ratio, 8)
        gate_layer = gate_layer or nn.Hardsigmoid
        activation_layer = activation_layer or nn.ReLU
        self.conv_reduce = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.act1 = activation_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def _se_or_skip(hidden_ch: int, input_ch: Optional[int], label: str) -> nn.LayerChoice:
    return nn.LayerChoice({
        'identity': nn.Identity(),
        'se': SqueezeExcite(hidden_ch)
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
                 bn_momentum: float = 0.1):
        super().__init__()

        assert len(base_widths) == 8
        assert len(width_multipliers) == 7

        self.widths = cast(nn.ChoiceOf[int], [
            nn.ValueChoice([make_divisible(base_width * mult, 8) for mult in width_multipliers], label=f's{i}_width')
            for i, base_width in enumerate(base_widths)
        ])
        self.expand_ratios = expand_ratios

        blocks = [
            # Stem
            ConvBNReLU(
                3, self.widths[0],
                nn.ValueChoice([3, 5], label=f'stem_ks'),
                stride=2, activation_layer=nn.Hardswish
            ),

            # Stage 0
            # FIXME: this should be an optional layer
            SeparableConv(
                self.widths[0], self.widths[0],
                nn.ValueChoice([3, 5, 7], label=f's0_i0_ks'),
                stride=1,
                activation_layer=nn.ReLU,
            ),
        ]

        blocks += [
            # Body: Stage 1-5
            self._make_stage(1, self.widths[0], self.widths[1], False, 2, nn.ReLU),
            self._make_stage(2, self.widths[1], self.widths[2], True, 2, nn.ReLU),
            self._make_stage(3, self.widths[2], self.widths[3], False, 2, nn.Hardswish),
            self._make_stage(4, self.widths[3], self.widths[4], True, 1, nn.Hardswish),
            self._make_stage(5, self.widths[4], self.widths[5], True, 2, nn.Hardswish),
        ]

        # NOTE: The built-in hardswish produces slightly different output from 3rd-party implementation
        # But I guess it doesn't really matter.
        # https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/layers/activations.py#L79

        # Head
        blocks += [
            ConvBNReLU(self.widths[5], self.widths[6], 1, 1, activation_layer=nn.Hardswish),
            nn.AdaptiveAvgPool2d(1),

            ConvBNReLU(self.widths[6], self.widths[7], 1, 1, norm_layer=nn.Identity, activation_layer=nn.Hardswish),
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
            )

        # mutable depth
        return nn.Repeat(layer_builder, depth=(1, 4), label=f's{stage_idx}_depth')
