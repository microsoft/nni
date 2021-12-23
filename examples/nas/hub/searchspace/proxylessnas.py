import functools
import math
import random
from typing import Optional, Callable, List

import torch
import torch.nn as nn

from nni.retiarii.serializer import model_wrapper


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
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
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

        layers.extend([
            # depth-wise
            ConvBNReLU(hidden_ch, hidden_ch, stride=stride, kernel_size=kernel_size, groups=hidden_ch, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_ch, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels),
        ])
        super().__init__(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            return x + super().forward(x)
        else:
            return super().forward(x)


@model_wrapper
class ProxylessSpace(nn.Module):
    """
    The search space proposed by ProxylessNAS [proxylessnas]_ .

    References
    ----------
    .. [proxylessnas] Cai, Han, Ligeng Zhu, and Song Han. "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware."
        International Conference on Learning Representations. 2018.
    """


class _MbNet(nn.Module):
    def __init__(self, first_conv, blocks, feature_mix_layer, dropout_layer, classifier):
        super().__init__()
        self.first_conv = first_conv
        self.blocks = nn.Sequential(*blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout_layer = dropout_layer
        self.classifier = classifier

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
        # no regularizer to linear layer
        return {'classifier.weight', 'classifier.bias'}

    def reset_parameters(self, model_init='he_fout', init_div_groups=False,
                         bn_momentum=0.1, bn_eps=1e-5,
                         track_running_stats=True, zero_grad=False):
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


class ProxylessNAS(_MbNet, SearchSpace):
    def __init__(self, config: ProxylessConfig, reset_parameters=True):
        stem_width = make_divisible(config.width_mult * config.stem_width, 8)

        first_conv = ConvBNReLU(3, stem_width, stride=2, norm_layer=nn.BatchNorm2d)

        last_width = stem_width
        blocks = []
        for i, stage_config in enumerate(config.stages, start=1):
            print_log(f'Building stage #{i}...', __name__)
            width = make_divisible(stage_config.width * config.width_mult, 8)
            blocks += self._build_stage(i, stage_config, last_width, width)
            last_width = width

        final_width = make_divisible(1280 * config.width_mult, 8) if config.width_mult > 1 else 1280
        dropout_layer = nn.Dropout(config.dropout_rate)
        feature_mix_layer = ConvBNReLU(last_width, final_width, kernel_size=1, norm_layer=nn.BatchNorm2d)
        classifier = nn.Linear(final_width, config.num_labels)
        super().__init__(first_conv, blocks, feature_mix_layer, dropout_layer, classifier)

        if reset_parameters:
            self.reset_parameters(track_running_stats=False, zero_grad=True)

    def _build_stage(self, stage_idx: int, config: ProxylessStageConfig, input_width: int, output_width: int):
        depth_min, depth_max = config.depth_range
        blocks = []
        for i in range(depth_max):
            stride = 2 if config.downsample and i == 0 else 1
            op_choices = {}
            for exp_ratio in config.exp_ratio_range:
                for kernel_size in config.kernel_size_range:
                    op_choices[f'k{kernel_size}e{exp_ratio}'] = InvertedResidual(input_width, output_width, stride, exp_ratio, kernel_size)
            if i >= depth_min:
                prior = [0.5 / len(op_choices)] * len(op_choices) + [0.5]
                op_choices['skip'] = nn.Identity()
                blocks.append(BiasedMixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices, prior))
                assert blocks[-1].op_candidates[-1] == 'skip'
            else:
                blocks.append(MixedOp(f's{stage_idx}b{i + 1}_i{input_width}o{output_width}', op_choices))
            print_log(f'Created block: {blocks[-1].key}: {blocks[-1].op_candidates}', __name__)
            input_width = output_width
        return blocks
