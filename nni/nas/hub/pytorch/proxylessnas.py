# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Optional, Callable, List, Tuple, Iterator, Union, cast, overload

import torch
from torch import nn
from nni.mutable import MutableExpression
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, Repeat, MutableConv2d, MutableLinear, MutableBatchNorm2d

from .utils.pretrained import load_pretrained_weight

MaybeIntChoice = Union[int, MutableExpression[int]]


@overload
def make_divisible(v: Union[int, float], divisor, min_val=None) -> int:
    ...


@overload
def make_divisible(v: Union[MutableExpression[int], MutableExpression[float]], divisor, min_val=None) -> MutableExpression[int]:
    ...


def make_divisible(v: Union[MutableExpression[int], MutableExpression[float], int, float], divisor, min_val=None) -> MaybeIntChoice:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    # This should work for both value choices and constants.
    new_v = MutableExpression.max(min_val, round(v + divisor // 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    return MutableExpression.condition(new_v < 0.9 * v, new_v + divisor, new_v)


def simplify_sequential(sequentials: List[nn.Module]) -> Iterator[nn.Module]:
    """
    Flatten the sequential blocks so that the hierarchy looks better.
    Eliminate identity modules automatically.
    """
    for module in sequentials:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                # no recursive expansion
                if not isinstance(submodule, nn.Identity):
                    yield submodule
        else:
            if not isinstance(module, nn.Identity):
                yield module


class ConvBNReLU(nn.Sequential):
    """
    The template for a conv-bn-relu block.
    """

    def __init__(
        self,
        in_channels: MaybeIntChoice,
        out_channels: MaybeIntChoice,
        kernel_size: MaybeIntChoice = 3,
        stride: int = 1,
        groups: MaybeIntChoice = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = MutableBatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        # If no normalization is used, set bias to True
        # https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L194
        norm = norm_layer(cast(int, out_channels))
        no_normalization = isinstance(norm, nn.Identity)
        blocks: List[nn.Module] = [
            MutableConv2d(
                cast(int, in_channels),
                cast(int, out_channels),
                cast(int, kernel_size),
                stride,
                cast(int, padding),
                dilation=dilation,
                groups=cast(int, groups),
                bias=no_normalization
            ),
            # Normalization, regardless of batchnorm or identity
            norm,
            # One pytorch implementation as an SE here, to faithfully reproduce paper
            # We follow a more accepted approach to put SE outside
            # Reference: https://github.com/d-li14/mobilenetv3.pytorch/issues/18
            activation_layer(inplace=True)
        ]

        super().__init__(*simplify_sequential(blocks))


class DepthwiseSeparableConv(nn.Sequential):
    """
    In the original MobileNetV2 implementation, this is InvertedResidual when expand ratio = 1.
    Residual connection is added if input and output shape are the same.

    References:

    - https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L90
    - https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L433
    - https://github.com/ultmaster/AceNAS/blob/46c8895f/searchspace/proxylessnas/utils.py#L100
    """

    def __init__(
        self,
        in_channels: MaybeIntChoice,
        out_channels: MaybeIntChoice,
        kernel_size: MaybeIntChoice = 3,
        stride: int = 1,
        squeeze_excite: Optional[Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        blocks = [
            # dw
            ConvBNReLU(in_channels, in_channels, stride=stride, kernel_size=kernel_size, groups=in_channels,
                       norm_layer=norm_layer, activation_layer=activation_layer),
            # optional se
            squeeze_excite(in_channels, in_channels) if squeeze_excite else nn.Identity(),
            # pw-linear
            ConvBNReLU(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity)
        ]
        super().__init__(*simplify_sequential(blocks))
        # NOTE: "is" is used here instead of "==" to avoid creating a new value choice.
        self.has_skip = stride == 1 and in_channels is out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + super().forward(x)
        else:
            return super().forward(x)


class InvertedResidual(nn.Sequential):
    """
    An Inverted Residual Block, sometimes called an MBConv Block, is a type of residual block used for image models
    that uses an inverted structure for efficiency reasons.

    It was originally proposed for the `MobileNetV2 <https://arxiv.org/abs/1801.04381>`__ CNN architecture.
    It has since been reused for several mobile-optimized CNNs.
    It follows a narrow -> wide -> narrow approach, hence the inversion.
    It first widens with a 1x1 convolution, then uses a 3x3 depthwise convolution (which greatly reduces the number of parameters),
    then a 1x1 convolution is used to reduce the number of channels so input and output can be added.

    This implementation is sort of a mixture between:

    - https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L453
    - https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L134

    Parameters
    ----------
    in_channels
        The number of input channels. Can be a value choice.
    out_channels
        The number of output channels. Can be a value choice.
    expand_ratio
        The ratio of intermediate channels with respect to input channels. Can be a value choice.
    kernel_size
        The kernel size of the depthwise convolution. Can be a value choice.
    stride
        The stride of the depthwise convolution.
    squeeze_excite
        Callable to create squeeze and excitation layer. Take hidden channels and input channels as arguments.
    norm_layer
        Callable to create normalization layer. Take input channels as argument.
    activation_layer
        Callable to create activation layer. No input arguments.
    """

    def __init__(
        self,
        in_channels: MaybeIntChoice,
        out_channels: MaybeIntChoice,
        expand_ratio: Union[float, MutableExpression[float]],
        kernel_size: MaybeIntChoice = 3,
        stride: int = 1,
        squeeze_excite: Optional[Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        assert stride in [1, 2]

        hidden_ch = cast(int, make_divisible(in_channels * expand_ratio, 8))

        # NOTE: this equivalence check (==) does NOT work for ValueChoice, need to use "is"
        self.has_skip = stride == 1 and in_channels is out_channels

        layers: List[nn.Module] = [
            # point-wise convolution
            # NOTE: some paper omit this point-wise convolution when stride = 1.
            # In our implementation, if this pw convolution is intended to be omitted,
            # please use SepConv instead.
            ConvBNReLU(in_channels, hidden_ch, kernel_size=1,
                       norm_layer=norm_layer, activation_layer=activation_layer),
            # depth-wise
            ConvBNReLU(hidden_ch, hidden_ch, stride=stride, kernel_size=kernel_size, groups=hidden_ch,
                       norm_layer=norm_layer, activation_layer=activation_layer),
            # SE
            squeeze_excite(
                cast(int, hidden_ch),
                cast(int, in_channels)
            ) if squeeze_excite is not None else nn.Identity(),
            # pw-linear
            ConvBNReLU(hidden_ch, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity),
        ]

        super().__init__(*simplify_sequential(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + super().forward(x)
        else:
            return super().forward(x)


def inverted_residual_choice_builder(
    expand_ratios: List[int],
    kernel_sizes: List[int],
    downsample: bool,
    stage_input_width: int,
    stage_output_width: int,
    label: str
):
    def builder(index):
        stride = 1
        inp = stage_output_width

        if index == 0:
            # first layer in stage
            # do downsample and width reshape
            inp = stage_input_width
            if downsample:
                stride = 2

        oup = stage_output_width

        op_choices = {}
        for exp_ratio in expand_ratios:
            for kernel_size in kernel_sizes:
                op_choices[f'k{kernel_size}e{exp_ratio}'] = InvertedResidual(inp, oup, exp_ratio, kernel_size, stride)

        # It can be implemented with ValueChoice, but we use LayerChoice here
        # to be aligned with the intention of the original ProxylessNAS.
        return LayerChoice(op_choices, label=f'{label}_i{index}')

    return builder


class ProxylessNAS(ModelSpace):
    """
    The search space proposed by `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__.

    Following the official implementation, the inverted residual with kernel size / expand ratio variations in each layer
    is implemented with a :class:`~nni.retiarii.nn.pytorch.LayerChoice` with all-combination candidates. That means,
    when used in weight sharing, these candidates will be treated as separate layers, and won't be fine-grained shared.
    We note that :class:`MobileNetV3Space` is different in this perspective.

    This space can be implemented as part of :class:`MobileNetV3Space`, but we separate those following conventions.

    Parameters
    ----------
    num_labels
        The number of labels for classification.
    base_widths
        Widths of each stage, from stem, to body, to head. Length should be 9.
    dropout_rate
        Dropout rate for the final classification layer.
    width_mult
        Width multiplier for the model.
    bn_eps
        Epsilon for batch normalization.
    bn_momentum
        Momentum for batch normalization.
    """

    def __init__(self, num_labels: int = 1000,
                 base_widths: Tuple[int, ...] = (32, 16, 32, 40, 80, 96, 192, 320, 1280),
                 dropout_rate: float = 0.,
                 width_mult: float = 1.0,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.1):

        super().__init__()

        assert len(base_widths) == 9
        # include the last stage info widths here
        widths = [make_divisible(width * width_mult, 8) for width in base_widths]
        downsamples = [True, False, True, True, True, False, True, False]

        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        self.stem = ConvBNReLU(3, widths[0], stride=2, norm_layer=MutableBatchNorm2d)

        blocks: List[nn.Module] = [
            # first stage is fixed
            DepthwiseSeparableConv(widths[0], widths[1], kernel_size=3, stride=1)
        ]

        # https://github.com/ultmaster/AceNAS/blob/46c8895fd8a05ffbc61a6b44f1e813f64b4f66b7/searchspace/proxylessnas/__init__.py#L21
        for stage in range(2, 8):
            # Rather than returning a fixed module here,
            # we return a builder that dynamically creates module for different `repeat_idx`.
            builder = inverted_residual_choice_builder(
                [3, 6], [3, 5, 7], downsamples[stage], widths[stage - 1], widths[stage], f's{stage}')
            if stage < 7:
                blocks.append(Repeat(builder, (1, 4), label=f's{stage}_depth'))
            else:
                # No mutation for depth in the last stage.
                # Directly call builder to initiate one block
                blocks.append(builder(0))

        self.blocks = nn.Sequential(*blocks)

        # final layers
        self.feature_mix_layer = ConvBNReLU(widths[7], widths[8], kernel_size=1, norm_layer=MutableBatchNorm2d)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.classifier = MutableLinear(widths[-1], num_labels)

        reset_parameters(self, bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout_layer(x)
        x = self.classifier(x)
        return x

    def no_weight_decay(self):
        # this is useful for timm optimizer
        # no regularizer to linear layer
        if hasattr(self, 'classifier'):
            return {'classifier.weight', 'classifier.bias'}
        return set()

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = False, progress: bool = True
    ) -> nn.Module:

        init_kwargs = {}  # all default

        if name == 'acenas-m1':
            arch = {
                's2_depth': 2,
                's2_i0': 'k3e6',
                's2_i1': 'k3e3',
                's3_depth': 3,
                's3_i0': 'k5e3',
                's3_i1': 'k3e3',
                's3_i2': 'k5e3',
                's4_depth': 2,
                's4_i0': 'k3e6',
                's4_i1': 'k5e3',
                's5_depth': 4,
                's5_i0': 'k7e6',
                's5_i1': 'k3e6',
                's5_i2': 'k3e6',
                's5_i3': 'k7e3',
                's6_depth': 4,
                's6_i0': 'k7e6',
                's6_i1': 'k7e6',
                's6_i2': 'k7e3',
                's6_i3': 'k7e3',
                's7_depth': 1,
                's7_i0': 'k7e6'
            }

        elif name == 'acenas-m2':
            arch = {
                's2_depth': 1,
                's2_i0': 'k5e3',
                's3_depth': 3,
                's3_i0': 'k3e6',
                's3_i1': 'k3e3',
                's3_i2': 'k5e3',
                's4_depth': 2,
                's4_i0': 'k7e6',
                's4_i1': 'k5e6',
                's5_depth': 4,
                's5_i0': 'k5e6',
                's5_i1': 'k5e3',
                's5_i2': 'k5e6',
                's5_i3': 'k3e6',
                's6_depth': 4,
                's6_i0': 'k7e6',
                's6_i1': 'k5e6',
                's6_i2': 'k5e3',
                's6_i3': 'k5e6',
                's7_depth': 1,
                's7_i0': 'k7e6'
            }

        elif name == 'acenas-m3':
            arch = {
                's2_depth': 2,
                's2_i0': 'k3e3',
                's2_i1': 'k3e6',
                's3_depth': 2,
                's3_i0': 'k5e3',
                's3_i1': 'k3e3',
                's4_depth': 3,
                's4_i0': 'k5e6',
                's4_i1': 'k7e6',
                's4_i2': 'k3e6',
                's5_depth': 4,
                's5_i0': 'k7e6',
                's5_i1': 'k7e3',
                's5_i2': 'k7e3',
                's5_i3': 'k5e3',
                's6_depth': 4,
                's6_i0': 'k7e6',
                's6_i1': 'k7e3',
                's6_i2': 'k7e6',
                's6_i3': 'k3e3',
                's7_depth': 1,
                's7_i0': 'k5e6'
            }

        elif name == 'proxyless-cpu':
            arch = {
                's2_depth': 4,
                's2_i0': 'k3e6',
                's2_i1': 'k3e3',
                's2_i2': 'k3e3',
                's2_i3': 'k3e3',
                's3_depth': 4,
                's3_i0': 'k3e6',
                's3_i1': 'k3e3',
                's3_i2': 'k3e3',
                's3_i3': 'k5e3',
                's4_depth': 2,
                's4_i0': 'k3e6',
                's4_i1': 'k3e3',
                's5_depth': 4,
                's5_i0': 'k5e6',
                's5_i1': 'k3e3',
                's5_i2': 'k3e3',
                's5_i3': 'k3e3',
                's6_depth': 4,
                's6_i0': 'k5e6',
                's6_i1': 'k5e3',
                's6_i2': 'k5e3',
                's6_i3': 'k3e3',
                's7_depth': 1,
                's7_i0': 'k5e6'
            }

            init_kwargs['base_widths'] = [40, 24, 32, 48, 88, 104, 216, 360, 1432]

        elif name == 'proxyless-gpu':
            arch = {
                's2_depth': 1,
                's2_i0': 'k5e3',
                's3_depth': 2,
                's3_i0': 'k7e3',
                's3_i1': 'k3e3',
                's4_depth': 2,
                's4_i0': 'k7e6',
                's4_i1': 'k5e3',
                's5_depth': 3,
                's5_i0': 'k5e6',
                's5_i1': 'k3e3',
                's5_i2': 'k5e3',
                's6_depth': 4,
                's6_i0': 'k7e6',
                's6_i1': 'k7e6',
                's6_i2': 'k7e6',
                's6_i3': 'k5e6',
                's7_depth': 1,
                's7_i0': 'k7e6'
            }

            init_kwargs['base_widths'] = [40, 24, 32, 56, 112, 128, 256, 432, 1728]

        elif name == 'proxyless-mobile':
            arch = {
                's2_depth': 2,
                's2_i0': 'k5e3',
                's2_i1': 'k3e3',
                's3_depth': 4,
                's3_i0': 'k7e3',
                's3_i1': 'k3e3',
                's3_i2': 'k5e3',
                's3_i3': 'k5e3',
                's4_depth': 4,
                's4_i0': 'k7e6',
                's4_i1': 'k5e3',
                's4_i2': 'k5e3',
                's4_i3': 'k5e3',
                's5_depth': 4,
                's5_i0': 'k5e6',
                's5_i1': 'k5e3',
                's5_i2': 'k5e3',
                's5_i3': 'k5e3',
                's6_depth': 4,
                's6_i0': 'k7e6',
                's6_i1': 'k7e6',
                's6_i2': 'k7e3',
                's6_i3': 'k7e3',
                's7_depth': 1,
                's7_i0': 'k7e6'
            }

        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.frozen_factory(arch)
        model = model_factory(**init_kwargs)

        if pretrained:
            weight_file = load_pretrained_weight(name, download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model


def reset_parameters(model, model_init='he_fout', init_div_groups=False,
                     bn_momentum=0.1, bn_eps=1e-5):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == 'he_fout':
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if init_div_groups:
                    n /= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif model_init == 'he_fin':
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                if init_div_groups:
                    n /= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
            else:
                raise NotImplementedError
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            m.momentum = bn_momentum
            m.eps = bn_eps
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
