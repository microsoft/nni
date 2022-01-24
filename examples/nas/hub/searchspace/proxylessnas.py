import math
from typing import Optional, Callable, List, Tuple

import torch
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    The template for a conv-bn-relu block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        )
        self.out_channels = out_channels


class SELayer(nn.Module):
    """Squeeze-and-excite layer."""

    def __init__(self,
                 channels: int,
                 reduction: int = 4,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        if activation_layer is None:
            activation_layer = nn.ReLU6
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


class InvertedResidual(nn.Sequential):
    """
    An Inverted Residual Block, sometimes called an MBConv Block, is a type of residual block used for image models
    that uses an inverted structure for efficiency reasons.
    It was originally proposed for the MobileNetV2 CNN architecture [mobilenetv2]_ .
    It has since been reused for several mobile-optimized CNNs.
    It follows a narrow -> wide -> narrow approach, hence the inversion.
    It first widens with a 1x1 convolution, then uses a 3x3 depthwise convolution (which greatly reduces the number of parameters),
    then a 1x1 convolution is used to reduce the number of channels so input and output can be added.

    References
    ----------
    .. [mobilenetv2] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        omit_expansion: bool = True,
        squeeze_and_excite: bool = False,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.out_channels = out_channels
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_ch = int(round(in_channels * expand_ratio))

        # Residual connection is added here stride = 1 and input channels and output channels are the same.
        self.residual_connection = self.stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []

        # if weight sharing is enabled, this expansion can not be omitted, otherwise weights won't be able to be shared.
        if expand_ratio != 1 or not omit_expansion:
            # point-wise convolution
            layers.append(ConvBNReLU(in_channels, hidden_ch, kernel_size=1, norm_layer=norm_layer))

        layers.append(
            # depth-wise
            ConvBNReLU(hidden_ch, hidden_ch, stride=stride, kernel_size=kernel_size, groups=hidden_ch, norm_layer=norm_layer)
        )

        if squeeze_and_excite:
            layers.append(SELayer(hidden_ch))

        layers += [
            # pw-linear
            nn.Conv2d(hidden_ch, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels),
        ]

        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
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
        if index == 0 and downsample:
            # first layer in stage
            # do downsample and width reshape
            stride, inp = 2, stage_input_width
        else:
            # otherwise keep shape
            stride, inp = 1, stage_input_width
        oup = stage_output_width

        op_choices = {}
        for exp_ratio in expand_ratios:
            for kernel_size in kernel_sizes:
                op_choices[f'k{kernel_size}e{exp_ratio}'] = InvertedResidual(inp, oup, stride, exp_ratio, kernel_size)

        return nn.LayerChoice(op_choices, label=f'{label}_i{index}')

    return builder


@model_wrapper
class ProxylessSpace(nn.Module):
    """
    The search space proposed by ProxylessNAS [proxylessnas]_ .

    References
    ----------
    .. [proxylessnas] Cai, Han, Ligeng Zhu, and Song Han. "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware."
        International Conference on Learning Representations. 2018.
    """

    def __init__(self, num_labels: int = 1000,
                 base_widths: Tuple[int] = (16, 32, 40, 80, 96, 192, 320, 1280),
                 dropout_rate: float = 0.,
                 stem_width: int = 32,
                 width_mult: float = 1.0,
                 bn_eps: float = 1e-3,
                 bn_momentum: float = 0.1):

        assert len(base_widths) == 8
        # include the last stage info widths here
        widths = [make_divisible(base_widths * width_mult, 8) for width in base_widths]
        downsamples = [False, True, True, True, False, True, False]

        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.stem_width = stem_width
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        self.first_conv = ConvBNReLU(3, stem_width, stride=2, norm_layer=nn.BatchNorm2d)

        blocks = []

        # https://github.com/ultmaster/AceNAS/blob/46c8895fd8a05ffbc61a6b44f1e813f64b4f66b7/searchspace/proxylessnas/__init__.py#L21
        for stage in range(7):
            if stage == 0:
                # first stage is fixed
                blocks.append(InvertedResidual(stem_width, widths[0], stride=1, expand_ratio=1, kernel_size=3))
            else:
                builder = inverted_residual_choice_builder(
                    [3, 6], [3, 5, 7], downsamples[stage], widths[stage - 1], widths[stage], f's{stage}')
                if stage < 6:
                    blocks.append(nn.Repeat(builder, (1, 4), label='s{stage}_depth'))
                else:
                    # not mutating depth for last stage
                    # therefore directly call builder to 
                    blocks.append(builder())

        self.blocks = nn.Sequential(*blocks)

        # final layers
        self.feature_mix_layer = ConvBNReLU(widths[6], widths[7], kernel_size=1, norm_layer=nn.BatchNorm2d)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(widths[-1], num_labels)

        self.reset_parameters(bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, x):
        x = self.first_conv(x)
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

    def reset_parameters(self, model_init='he_fout', init_div_groups=False,
                         bn_momentum=0.1, bn_eps=1e-5):
        for m in self.modules():
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
