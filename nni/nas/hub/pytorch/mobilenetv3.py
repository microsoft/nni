# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Tuple, Optional, Callable, Union, List, Type, cast
from typing_extensions import Literal

import torch
from torch import nn

import nni
from nni.nas.nn.pytorch import ModelSpace, Repeat, LayerChoice, MutableLinear, MutableConv2d

from .proxylessnas import ConvBNReLU, InvertedResidual, DepthwiseSeparableConv, MaybeIntChoice, make_divisible, reset_parameters
from .utils.pretrained import load_pretrained_weight


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
        self.conv_reduce = MutableConv2d(channels, rd_channels, 1, bias=True)
        self.act1 = activation_layer(inplace=True)
        self.conv_expand = MutableConv2d(rd_channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def _se_or_skip(hidden_ch: int, input_ch: int, optional: bool, se_from_exp: bool, label: str) -> nn.Module:
    ch = hidden_ch if se_from_exp else input_ch
    if optional:
        return LayerChoice({
            'identity': nn.Identity(),
            'se': SqueezeExcite(ch)
        }, label=label)
    else:
        return SqueezeExcite(ch)


def _act_fn(act_alias: Literal['hswish', 'swish', 'relu']) -> Type[nn.Module]:
    if act_alias == 'hswish':
        return nn.Hardswish
    elif act_alias == 'swish':
        return nn.SiLU
    elif act_alias == 'relu':
        return nn.ReLU
    else:
        raise ValueError(f'Unsupported act alias: {act_alias}')


class MobileNetV3Space(ModelSpace):
    """
    MobileNetV3Space implements the largest search space in `TuNAS <https://arxiv.org/abs/2008.06120>`__.

    The search dimensions include widths, expand ratios, kernel sizes, SE ratio.
    Some of them can be turned off via arguments to narrow down the search space.

    Different from ProxylessNAS search space, this space is implemented with :class:`~nni.nas.nn.pytorch.ValueChoice`.

    We use the following snipppet as reference.
    https://github.com/google-research/google-research/blob/20736344591f774f4b1570af64624ed1e18d2867/tunas/mobile_search_space_v3.py#L728

    We have ``num_blocks`` which equals to the length of ``self.blocks`` (the main body of the network).
    For simplicity, the following parameter specification assumes ``num_blocks`` equals 8 (body + head).
    If a shallower body is intended, arrays including ``base_widths``, ``squeeze_excite``, ``depth_range``,
    ``stride``, ``activation`` should also be shortened accordingly.

    Parameters
    ----------
    num_labels
        Dimensions for classification head.
    base_widths
        Widths of each stage, from stem, to body, to head.
        Length should be 9, i.e., ``num_blocks + 1`` (because there is a stem width in front).
    width_multipliers
        A range of widths multiplier to choose from. The choice is independent for each stage.
        Or it can be a fixed float. This will be applied on ``base_widths``,
        and we would also make sure that widths can be divided by 8.
    expand_ratios
        A list of expand ratios to choose from. Independent for every **block**.
    squeeze_excite
        Indicating whether the current stage can have an optional SE layer.
        Expect array of length 6 for stage 0 to 5. Each element can be one of ``force``, ``optional``, ``none``.
    depth_range
        A range (e.g., ``(1, 4)``),
        or a list of range (e.g., ``[(1, 3), (1, 4), (1, 4), (1, 3), (0, 2)]``).
        If a list, the length should be 5. The depth are specified for stage 1 to 5.
    stride
        Stride for all stages (including stem and head). Length should be same as ``base_widths``.
    activation
        Activation (class) for all stages. Length is same as ``base_widths``.
    se_from_exp
        Calculate SE channel reduction from expanded (mid) channels.
    dropout_rate
        Dropout rate at classification head.
    bn_eps
        Epsilon of batch normalization.
    bn_momentum
        Momentum of batch normalization.
    """

    widths: List[MaybeIntChoice]
    depth_range: List[Tuple[int, int]]

    def __init__(
        self, num_labels: int = 1000,
        base_widths: Tuple[int, ...] = (16, 16, 16, 32, 64, 128, 256, 512, 1024),
        width_multipliers: Union[Tuple[float, ...], float] = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
        expand_ratios: Tuple[float, ...] = (1., 2., 3., 4., 5., 6.),
        squeeze_excite: Tuple[Literal['force', 'optional', 'none'], ...] = (
            'none', 'none', 'optional', 'none', 'optional', 'optional'
        ),
        depth_range: Union[List[Tuple[int, int]], Tuple[int, int]] = (1, 4),
        stride: Tuple[int, ...] = (2, 1, 2, 2, 2, 1, 2, 1, 1),
        activation: Tuple[Literal['hswish', 'swish', 'relu'], ...] = (
            'hswish', 'relu', 'relu', 'relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'
        ),
        se_from_exp: bool = True,
        dropout_rate: float = 0.2,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.1
    ):
        super().__init__()

        self.num_blocks = len(base_widths) - 1  # without stem, equal to len(self.blocks)
        assert self.num_blocks >= 4

        assert len(base_widths) == len(stride) == len(activation) == self.num_blocks + 1

        # The final two blocks can't have SE
        assert len(squeeze_excite) == self.num_blocks - 2 and all(se in ['force', 'optional', 'none'] for se in squeeze_excite)

        # The first and final two blocks can't have variational depth
        if isinstance(depth_range[0], int):
            depth_range = cast(Tuple[int, int], depth_range)
            assert len(depth_range) == 2 and depth_range[1] >= depth_range[0] >= 1
            self.depth_range = [depth_range] * (self.num_blocks - 3)
        else:
            assert len(depth_range) == self.num_blocks - 3
            self.depth_range = cast(List[Tuple[int, int]], depth_range)
            for d in self.depth_range:
                d = cast(Tuple[int, int], d)
                # pylint: disable=unsubscriptable-object
                assert len(d) == 2 and d[1] >= d[0] >= 1, f'{d} does not satisfy depth constraints'

        self.widths = []
        for i, base_width in enumerate(base_widths):
            if isinstance(width_multipliers, float):
                self.widths.append(make_divisible(base_width * width_multipliers, 8))
            else:
                self.widths.append(
                    # According to tunas, stem and stage 0 share one width multiplier
                    # https://github.com/google-research/google-research/blob/20736344/tunas/mobile_search_space_v3.py#L791
                    make_divisible(
                        nni.choice(f's{max(i - 1, 0)}_width_mult', list(width_multipliers)) * base_width, 8
                    )
                )

        self.expand_ratios = expand_ratios
        self.se_from_exp = se_from_exp

        # NOTE: The built-in hardswish produces slightly different output from 3rd-party implementation
        # But I guess it doesn't really matter.
        # https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/layers/activations.py#L79

        self.stem = ConvBNReLU(
            3, self.widths[0],
            nni.choice(f'stem_ks', [3, 5]),
            stride=stride[0], activation_layer=_act_fn(activation[0])
        )

        blocks: List[nn.Module] = [
            # Stage 0
            # FIXME: this should be an optional layer.
            # https://github.com/google-research/google-research/blob/20736344/tunas/mobile_search_space_v3.py#L791
            DepthwiseSeparableConv(
                self.widths[0], self.widths[1],
                nni.choice(f's0_i0_ks', [3, 5, 7]),
                stride=stride[1],
                squeeze_excite=cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], partial(
                    _se_or_skip, optional=squeeze_excite[0] == 'optional', se_from_exp=self.se_from_exp, label=f's0_i0_se'
                )) if squeeze_excite[0] != 'none' else None,
                activation_layer=_act_fn(activation[1])
            ),
        ]

        blocks += [
            # Stage 1-5 (by default)
            self._make_stage(i, self.widths[i], self.widths[i + 1], squeeze_excite[i], stride[i + 1], _act_fn(activation[i + 1]))
            for i in range(1, self.num_blocks - 2)
        ]

        # Head
        blocks += [
            ConvBNReLU(
                self.widths[self.num_blocks - 2],
                self.widths[self.num_blocks - 1],
                kernel_size=1,
                stride=stride[self.num_blocks - 1],
                activation_layer=_act_fn(activation[self.num_blocks - 1])
            ),
            nn.AdaptiveAvgPool2d(1),

            # In some implementation, this is a linear instead.
            # Should be equivalent.
            ConvBNReLU(
                self.widths[self.num_blocks - 1],
                self.widths[self.num_blocks],
                kernel_size=1,
                stride=stride[self.num_blocks],
                norm_layer=nn.Identity,
                activation_layer=_act_fn(activation[self.num_blocks])
            )
        ]

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            MutableLinear(cast(int, self.widths[self.num_blocks]), num_labels),
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
            exp = nni.choice(f's{stage_idx}_i{idx}_exp', list(self.expand_ratios))
            ks = nni.choice(f's{stage_idx}_i{idx}_ks', [3, 5, 7])
            # if SE is true, assign a layer choice to SE
            se_or_skip = cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], partial(
                _se_or_skip, optional=se == 'optional', se_from_exp=self.se_from_exp, label=f's{stage_idx}_i{idx}_se'
            )) if se != 'none' else None
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
        return Repeat(layer_builder, depth=(min_depth, max_depth), label=f's{stage_idx}_depth')

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = False, progress: bool = True
    ) -> nn.Module:

        init_kwargs = {}  # all default

        if name == 'mobilenetv3-large-100':
            # NOTE: Use bicsubic interpolation to evaluate this
            # With default interpolation, it yields top-1 75.722
            arch = {
                'stem_ks': 3,
                's0_i0_ks': 3,
                's1_depth': 2,
                's1_i0_exp': 4,
                's1_i0_ks': 3,
                's1_i1_exp': 3,
                's1_i1_ks': 3,
                's2_depth': 3,
                's2_i0_exp': 3,
                's2_i0_ks': 5,
                's2_i1_exp': 3,
                's2_i1_ks': 5,
                's2_i2_exp': 3,
                's2_i2_ks': 5,
                's3_depth': 4,
                's3_i0_exp': 6,
                's3_i0_ks': 3,
                's3_i1_exp': 2.5,
                's3_i1_ks': 3,
                's3_i2_exp': 2.3,
                's3_i2_ks': 3,
                's3_i3_exp': 2.3,
                's3_i3_ks': 3,
                's4_depth': 2,
                's4_i0_exp': 6,
                's4_i0_ks': 3,
                's4_i1_exp': 6,
                's4_i1_ks': 3,
                's5_depth': 3,
                's5_i0_exp': 6,
                's5_i0_ks': 5,
                's5_i1_exp': 6,
                's5_i1_ks': 5,
                's5_i2_exp': 6,
                's5_i2_ks': 5,
            }

            init_kwargs.update(
                base_widths=[16, 16, 24, 40, 80, 112, 160, 960, 1280],
                expand_ratios=[1.0, 2.0, 2.3, 2.5, 3.0, 4.0, 6.0],
                bn_eps=1e-5,
                bn_momentum=0.1,
                width_multipliers=1.0,
                squeeze_excite=['none', 'none', 'force', 'none', 'force', 'force']
            )

        elif name.startswith('mobilenetv3-small-'):
            # Evaluate with bicubic interpolation
            multiplier = int(name.split('-')[-1]) / 100
            widths = [16, 16, 24, 40, 48, 96, 576, 1024]
            for i in range(7):
                if i > 0 or multiplier >= 0.75:
                    # fix_stem = True when multiplier < 0.75
                    # https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/mobilenetv3.py#L421
                    widths[i] = make_divisible(widths[i] * multiplier, 8)
            init_kwargs.update(
                base_widths=widths,
                width_multipliers=1.0,
                expand_ratios=[3.0, 3.67, 4.0, 4.5, 6.0],
                bn_eps=1e-05,
                bn_momentum=0.1,
                squeeze_excite=['force', 'none', 'force', 'force', 'force'],
                activation=['hswish', 'relu', 'relu', 'hswish', 'hswish', 'hswish', 'hswish', 'hswish'],
                stride=[2, 2, 2, 2, 1, 2, 1, 1],
                depth_range=(1, 2),
            )

            arch = {
                'stem_ks': 3,
                's0_i0_ks': 3,
                's1_depth': 2,
                's1_i0_exp': 4.5,
                's1_i0_ks': 3,
                's1_i1_exp': 3.67,
                's1_i1_ks': 3,
                's2_depth': 3,
                's2_i0_exp': 4.0,
                's2_i0_ks': 5,
                's2_i1_exp': 6.0,
                's2_i1_ks': 5,
                's2_i2_exp': 6.0,
                's2_i2_ks': 5,
                's3_depth': 2,
                's3_i0_exp': 3.0,
                's3_i0_ks': 5,
                's3_i1_exp': 3.0,
                's3_i1_ks': 5,
                's4_depth': 3,
                's4_i0_exp': 6.0,
                's4_i0_ks': 5,
                's4_i1_exp': 6.0,
                's4_i1_ks': 5,
                's4_i2_exp': 6.0,
                's4_i2_ks': 5
            }

        elif name.startswith('cream'):
            # https://github.com/microsoft/Cream/tree/main/Cream
            # bilinear interpolation

            level = name.split('-')[-1]

            # region cream arch specification
            if level == '014':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 1,
                    's1_i0_exp': 4.0,
                    's1_i0_ks': 3,
                    's2_depth': 2,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 6.0,
                    's2_i1_ks': 5,
                    's3_depth': 2,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 6.0,
                    's3_i1_ks': 5,
                    's4_depth': 1,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 3,
                    's5_depth': 1,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5
                }
            elif level == '043':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 1,
                    's1_i0_exp': 4.0,
                    's1_i0_ks': 3,
                    's2_depth': 2,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 6.0,
                    's2_i1_ks': 3,
                    's3_depth': 2,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 6.0,
                    's3_i1_ks': 3,
                    's4_depth': 3,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 5,
                    's4_i1_exp': 6.0,
                    's4_i1_ks': 5,
                    's4_i2_exp': 6.0,
                    's4_i2_ks': 5,
                    's5_depth': 2,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5,
                    's5_i1_exp': 6.0,
                    's5_i1_ks': 5
                }
            elif level == '114':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 1,
                    's1_i0_exp': 4.0,
                    's1_i0_ks': 3,
                    's2_depth': 2,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 6.0,
                    's2_i1_ks': 5,
                    's3_depth': 2,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 6.0,
                    's3_i1_ks': 5,
                    's4_depth': 3,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 5,
                    's4_i1_exp': 6.0,
                    's4_i1_ks': 5,
                    's4_i2_exp': 6.0,
                    's4_i2_ks': 5,
                    's5_depth': 2,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5,
                    's5_i1_exp': 6.0,
                    's5_i1_ks': 5
                }
            elif level == '287':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 1,
                    's1_i0_exp': 4.0,
                    's1_i0_ks': 3,
                    's2_depth': 2,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 6.0,
                    's2_i1_ks': 5,
                    's3_depth': 3,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 6.0,
                    's3_i1_ks': 3,
                    's3_i2_exp': 6.0,
                    's3_i2_ks': 5,
                    's4_depth': 4,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 5,
                    's4_i1_exp': 6.0,
                    's4_i1_ks': 5,
                    's4_i2_exp': 6.0,
                    's4_i2_ks': 5,
                    's4_i3_exp': 6.0,
                    's4_i3_ks': 5,
                    's5_depth': 3,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5,
                    's5_i1_exp': 6.0,
                    's5_i1_ks': 5,
                    's5_i2_exp': 6.0,
                    's5_i2_ks': 5
                }
            elif level == '481':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 4,
                    's1_i0_exp': 6.0,
                    's1_i0_ks': 5,
                    's1_i1_exp': 4.0,
                    's1_i1_ks': 7,
                    's1_i2_exp': 6.0,
                    's1_i2_ks': 5,
                    's1_i3_exp': 6.0,
                    's1_i3_ks': 3,
                    's2_depth': 4,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 4.0,
                    's2_i1_ks': 5,
                    's2_i2_exp': 6.0,
                    's2_i2_ks': 5,
                    's2_i3_exp': 4.0,
                    's2_i3_ks': 3,
                    's3_depth': 5,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 6.0,
                    's3_i1_ks': 5,
                    's3_i2_exp': 6.0,
                    's3_i2_ks': 5,
                    's3_i3_exp': 6.0,
                    's3_i3_ks': 3,
                    's3_i4_exp': 6.0,
                    's3_i4_ks': 3,
                    's4_depth': 4,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 5,
                    's4_i1_exp': 6.0,
                    's4_i1_ks': 5,
                    's4_i2_exp': 6.0,
                    's4_i2_ks': 5,
                    's4_i3_exp': 6.0,
                    's4_i3_ks': 5,
                    's5_depth': 4,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5,
                    's5_i1_exp': 6.0,
                    's5_i1_ks': 5,
                    's5_i2_exp': 6.0,
                    's5_i2_ks': 5,
                    's5_i3_exp': 6.0,
                    's5_i3_ks': 5
                }
            elif level == '604':
                arch = {
                    'stem_ks': 3,
                    's0_depth': 1,
                    's0_i0_ks': 3,
                    's1_depth': 5,
                    's1_i0_exp': 6.0,
                    's1_i0_ks': 5,
                    's1_i1_exp': 6.0,
                    's1_i1_ks': 5,
                    's1_i2_exp': 4.0,
                    's1_i2_ks': 5,
                    's1_i3_exp': 6.0,
                    's1_i3_ks': 5,
                    's1_i4_exp': 6.0,
                    's1_i4_ks': 5,
                    's2_depth': 5,
                    's2_i0_exp': 6.0,
                    's2_i0_ks': 5,
                    's2_i1_exp': 4.0,
                    's2_i1_ks': 5,
                    's2_i2_exp': 6.0,
                    's2_i2_ks': 5,
                    's2_i3_exp': 4.0,
                    's2_i3_ks': 5,
                    's2_i4_exp': 6.0,
                    's2_i4_ks': 5,
                    's3_depth': 5,
                    's3_i0_exp': 6.0,
                    's3_i0_ks': 5,
                    's3_i1_exp': 4.0,
                    's3_i1_ks': 5,
                    's3_i2_exp': 6.0,
                    's3_i2_ks': 5,
                    's3_i3_exp': 4.0,
                    's3_i3_ks': 5,
                    's3_i4_exp': 6.0,
                    's3_i4_ks': 5,
                    's4_depth': 6,
                    's4_i0_exp': 6.0,
                    's4_i0_ks': 5,
                    's4_i1_exp': 6.0,
                    's4_i1_ks': 5,
                    's4_i2_exp': 4.0,
                    's4_i2_ks': 5,
                    's4_i3_exp': 4.0,
                    's4_i3_ks': 5,
                    's4_i4_exp': 6.0,
                    's4_i4_ks': 5,
                    's4_i5_exp': 6.0,
                    's4_i5_ks': 5,
                    's5_depth': 6,
                    's5_i0_exp': 6.0,
                    's5_i0_ks': 5,
                    's5_i1_exp': 6.0,
                    's5_i1_ks': 5,
                    's5_i2_exp': 4.0,
                    's5_i2_ks': 5,
                    's5_i3_exp': 6.0,
                    's5_i3_ks': 5,
                    's5_i4_exp': 6.0,
                    's5_i4_ks': 5,
                    's5_i5_exp': 6.0,
                    's5_i5_ks': 5
                }
            else:
                raise ValueError(f'Unsupported cream model level: {level}')
            # endregion

            init_kwargs.update(
                base_widths=[16, 16, 24, 40, 80, 96, 192, 320, 1280],
                width_multipliers=1.0,
                expand_ratios=[4.0, 6.0],
                bn_eps=1e-5,
                bn_momentum=0.1,
                squeeze_excite=['force'] * 6,
                activation=['swish'] * 9
            )

        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.frozen_factory(arch)
        model = model_factory(**init_kwargs)

        if pretrained:
            weight_file = load_pretrained_weight(name, download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model
