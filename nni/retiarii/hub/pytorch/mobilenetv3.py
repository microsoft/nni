# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Tuple, Optional, Callable, Union, List, cast

import torch.nn.functional as F
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper

from .proxylessnas import ConvBNReLU, InvertedResidual, DepthwiseSeparableConv, make_divisible, reset_parameters


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


def _se_or_skip(hidden_ch: int, input_ch: int, label: str) -> nn.LayerChoice:
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

    Parameters
    ----------
    num_labels
        Dimensions for classification head.
    base_widths
        Widths of each stage, from stem, to body, to head.
    width_multipliers
        A range of widths multiplier to choose from. The choice is independent for each stage.
        Or it can be a fixed float. This will be applied on ``base_widths``,
        and we would also make sure that widths can be divided by 8.
    expand_ratios
        A list of expand ratios to choose from. Independent for every **block**.
    squeeze_excite
        Indicating whether the current stage can have an optional SE layer.
        Expect boolean array of length 6 for stage 0 to 5.
    depth_range
        A range (e.g., ``(1, 4)``),
        or a list of range (e.g., ``[(1, 3), (1, 4), (1, 4), (1, 3), (0, 2)]``).
        If a list, the length should be 5. The depth are specified for stage 1 to 5.
    dropout_rate
        Dropout rate at classification head.
    bn_eps
        Epsilon of batch normalization.
    bn_momentum
        Momentum of batch normalization.
    """

    widths: List[Union[nn.ChoiceOf[int], int]]
    depth_range: List[Tuple[int, int]]

    def __init__(self, num_labels: int = 1000,
                 base_widths: Tuple[int, ...] = (16, 16, 16, 32, 64, 128, 256, 512, 1024),
                 width_multipliers: Union[Tuple[float, ...], float] = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
                 expand_ratios: Tuple[float, ...] = (1., 2., 3., 4., 5., 6.),
                 squeeze_excite: Tuple[bool, ...] = (False, False, True, False, True, True),
                 depth_range: Union[List[Tuple[int, int]], Tuple[int, int]] = (1, 4),
                 dropout_rate: float = 0.2,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.1):
        super().__init__()

        assert len(base_widths) == 9
        assert len(width_multipliers) == 7
        assert len(squeeze_excite) == 6
        if isinstance(depth_range[0], int):
            assert len(depth_range) == 2 and depth_range[1] >= depth_range[0] >= 0 and depth_range[1] >= 1
            self.depth_range = [depth_range] * 5
        else:
            assert len(depth_range) == 5
            self.depth_range = depth_range
            for d in self.depth_range:
                assert len(d) == 2 and d[1] >= d[0] >= 0 and d[1] >= 1, f"{d} does not satisfy depth constraints"

        self.widths = []
        for i, base_width in enumerate(base_widths):
            if isinstance(width_multipliers, float):
                self.widths.append(make_divisible(base_width * width_multipliers, 8))
            else:
                self.widths.append(
                    # According to tunas, stem and stage 0 share one width multiplier
                    # https://github.com/google-research/google-research/blob/20736344/tunas/mobile_search_space_v3.py#L791
                    make_divisible(
                        nn.ValueChoice(width_multipliers, label=f's{max(i - 1, 0)}_width_mult') * base_width, 8
                    )
                )

        self.expand_ratios = expand_ratios

        # NOTE: The built-in hardswish produces slightly different output from 3rd-party implementation
        # But I guess it doesn't really matter.
        # https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/layers/activations.py#L79

        self.stem = ConvBNReLU(
            3, self.widths[0],
            nn.ValueChoice([3, 5], label=f'stem_ks'),
            stride=2, activation_layer=nn.Hardswish
        )

        blocks = [
            # Stage 0
            # FIXME: this should be an optional layer.
            # https://github.com/google-research/google-research/blob/20736344/tunas/mobile_search_space_v3.py#L791
            DepthwiseSeparableConv(
                self.widths[0], self.widths[1],
                nn.ValueChoice([3, 5, 7], label=f's0_i0_ks'),
                stride=1,
                squeeze_excite=partial(_se_or_skip, label=f's0_i0_se') if squeeze_excite[0] else None,
                activation_layer=nn.ReLU,
            ),
        ]

        blocks += [
            # Stage 1-5
            self._make_stage(1, self.widths[1], self.widths[2], squeeze_excite[1], 2, nn.ReLU),
            self._make_stage(2, self.widths[2], self.widths[3], squeeze_excite[2], 2, nn.ReLU),
            self._make_stage(3, self.widths[3], self.widths[4], squeeze_excite[3], 2, nn.Hardswish),
            self._make_stage(4, self.widths[4], self.widths[5], squeeze_excite[4], 1, nn.Hardswish),
            self._make_stage(5, self.widths[5], self.widths[6], squeeze_excite[5], 2, nn.Hardswish),
        ]

        # Head
        blocks += [
            ConvBNReLU(self.widths[6], self.widths[7], 1, 1, activation_layer=nn.Hardswish),
            nn.AdaptiveAvgPool2d(1),

            # In some implementation, this is a linear instead.
            # Should be equivalent.
            ConvBNReLU(self.widths[7], self.widths[8], 1, 1, norm_layer=nn.Identity, activation_layer=nn.Hardswish),
        ]

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(cast(int, self.widths[8]), num_labels),
        )

        reset_parameters(self, bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_stage(self, stage_idx, inp, oup, se, stride, act):
        def layer_builder(idx):
            exp = nn.ValueChoice(list(self.expand_ratios), label=f's{stage_idx}_i{idx}_exp')
            ks = nn.ValueChoice([3, 5, 7], label=f's{stage_idx}_i{idx}_ks')
            # if SE is true, assign a layer choice to SE
            se_or_skip = partial(_se_or_skip, label=f's{stage_idx}_i{idx}_se') if se else None
            return InvertedResidual(
                inp if idx == 0 else oup,
                oup, exp, ks,
                stride=stride if idx == 0 else 1,  # only the first layer in each stage can have stride > 1
                squeeze_excite=se_or_skip,
                activation_layer=act,
            )

        # mutable depth
        min_depth, max_depth = self.depth_range[stage_idx - 1]
        if stride != 1:
            min_depth = max(min_depth, 1)
        return nn.Repeat(layer_builder, depth=(min_depth, max_depth), label=f's{stage_idx}_depth')
