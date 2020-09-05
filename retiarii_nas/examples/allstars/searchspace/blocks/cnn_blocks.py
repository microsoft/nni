import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pool(nn.Sequential):
    def __init__(self, pool_type, in_channels, out_channels, stride):
        super(Pool, self).__init__()
        if pool_type == "max":
            self.pool = nn.MaxPool2d(3, stride=stride, padding=1)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class Identity(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if stride == 1 and in_channels == out_channels:
            self.identity = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride)
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)


class BranchSeparables(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, stem=False, bias=False):
        super(BranchSeparables, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels if stem else in_channels
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, mid_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(mid_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


class StdConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(StdConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3x3BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class Conv1x1BnRelu(ConvBnRelu):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


class MaxPool3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPool3x3, self).__init__()
        self.maxpool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        out = self.maxpool(x)
        return out


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=True, momentum=0.1,
                           track_running_stats=True)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=True, momentum=0.1,
                           track_running_stats=True),
        )

    def forward(self, x):
        return self.op(x)


class Pooling(nn.Module):

    def __init__(self, C_in, C_out, stride, mode):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0)
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in Pooling".format(mode))

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=True, momentum=0.1,
                                 track_running_stats=True)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ResNetBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)  # residual
        return inputs + basicblock
