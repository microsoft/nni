# This file is downloaded from
# https://github.com/rwightman/pytorch-image-models

import torch.nn as nn

from timm.models.layers import create_conv2d
from timm.models.efficientnet_blocks import make_divisible, resolve_se_args, \
    SqueezeExcite, drop_path


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(
            self,
            in_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            dilation=1,
            pad_type='',
            act_layer=nn.ReLU,
            noskip=False,
            exp_ratio=1.0,
            exp_kernel_size=1,
            pw_kernel_size=1,
            se_ratio=0.,
            se_kwargs=None,
            norm_layer=nn.BatchNorm2d,
            norm_kwargs=None,
            conv_kwargs=None,
            drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(
            in_chs,
            mid_chs,
            exp_kernel_size,
            padding=pad_type,
            **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            mid_chs,
            out_chs,
            pw_kernel_size,
            padding=pad_type,
            **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            info = dict(
                module='conv_pwl',
                hook_type='forward_pre',
                num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            info = dict(
                module='',
                hook_type='',
                num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual

        return x
