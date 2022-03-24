# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch.nn as nn
from nni.retiarii import model_wrapper
from nni.retiarii.nn.pytorch import NasBench101Cell


__all__ = ['NasBench101']


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                truncated_normal_(m.weight.data, mean=0., std=math.sqrt(1. / fan_in))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.conv_bn_relu(x)


class Conv3x3BNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BNReLU, self).__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class Conv1x1BNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1BNReLU, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


Projection = Conv1x1BNReLU


@model_wrapper
class NasBench101(nn.Module):
    """The full search space, proposed by `NAS-Bench-101 <http://proceedings.mlr.press/v97/ying19a/ying19a.pdf>`__.

    It's simply a stack of :class:`NasBench101Cell`. Operations are conv3x3, conv1x1 and maxpool respectively.
    """

    def __init__(self,
                 stem_out_channels: int = 128,
                 num_stacks: int = 3,
                 num_modules_per_stack: int = 3,
                 max_num_vertices: int = 7,
                 max_num_edges: int = 9,
                 num_labels: int = 10,
                 bn_eps: float = 1e-5,
                 bn_momentum: float = 0.003):
        super().__init__()

        op_candidates = {
            'conv3x3-bn-relu': lambda num_features: Conv3x3BNReLU(num_features, num_features),
            'conv1x1-bn-relu': lambda num_features: Conv1x1BNReLU(num_features, num_features),
            'maxpool3x3': lambda num_features: nn.MaxPool2d(3, 1, 1)
        }

        # initial stem convolution
        self.stem_conv = Conv3x3BNReLU(3, stem_out_channels)

        layers = []
        in_channels = out_channels = stem_out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(num_modules_per_stack):
                cell = NasBench101Cell(op_candidates, in_channels, out_channels,
                                       lambda cin, cout: Projection(cin, cout),
                                       max_num_vertices, max_num_edges, label='cell')
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_labels)

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = bn_eps
                module.momentum = bn_momentum

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.classifier(out)
        return out

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = self.config.bn_eps
                module.momentum = self.config.bn_momentum
