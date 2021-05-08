# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch

import torch.nn as nn
import torch.nn.functional as F


# Basic primitives as the network path
PRIMITIVES = {
    "skip": lambda c_in, c_out, stride, **kwargs: Identity(
        c_in, c_out, stride, **kwargs
    ),
    "conv1x1": lambda c_in, c_out, stride, **kwargs: Conv1x1(
        c_in, c_out, stride, **kwargs
    ),
    "depth_conv": lambda c_in, c_out, stride, **kwargs: DepthConv(
        c_in, c_out, stride, **kwargs
    ),
    "sep_k3": lambda c_in, c_out, stride, **kwargs: SeparableConv(
        c_in, c_out, stride, **kwargs
    ),
    "sep_k5": lambda c_in, c_out, stride, **kwargs: SeparableConv(
        c_in, c_out, stride, kernel=5, **kwargs
    ),
    "gh_k3": lambda c_in, c_out, stride, **kwargs: GhostModule(
        c_in, c_out, stride, **kwargs
    ),
    "gh_k5": lambda c_in, c_out, stride, **kwargs: GhostModule(
        c_in, c_out, stride, kernel=5, **kwargs
    ),
    "mb_k3": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=1, **kwargs
    ),
    "mb_k3_e2": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=2, **kwargs
    ),
    "mb_k3_e4": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=4, **kwargs
    ),
    "mb_k3_res": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=1, res=True, **kwargs
    ),
    "mb_k3_e2_res": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=2, res=True, **kwargs
    ),
    "mb_k3_e4_res": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=3, expand=4, res=True, **kwargs
    ),
    "mb_k3_d2": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=2,
        res=False,
        dilation=2,
        **kwargs,
    ),
    "mb_k3_d3": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=2,
        res=False,
        dilation=3,
        **kwargs,
    ),
    "mb_k3_res_d2": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=2,
        res=True,
        dilation=2,
        **kwargs,
    ),
    "mb_k3_res_d3": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=2,
        res=True,
        dilation=3,
        **kwargs,
    ),
    "mb_k3_res_se": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=1,
        res=True,
        dilation=1,
        se=True,
        **kwargs,
    ),
    "mb_k3_e2_res_se": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=2,
        res=True,
        dilation=1,
        se=True,
        **kwargs,
    ),
    "mb_k3_e4_res_se": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=3,
        expand=4,
        res=True,
        dilation=1,
        se=True,
        **kwargs,
    ),
    "mb_k5": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=5, expand=1, **kwargs
    ),
    "mb_k5_e2": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=5, expand=2, **kwargs
    ),
    "mb_k5_res": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=5, expand=1, res=True, **kwargs
    ),
    "mb_k5_e2_res": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in, c_out, stride, kernel=5, expand=2, res=True, **kwargs
    ),
    "mb_k5_res_se": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=5,
        expand=1,
        res=True,
        dilation=1,
        se=True,
        **kwargs,
    ),
    "mb_k5_e2_res_se": lambda c_in, c_out, stride, **kwargs: MBBlock(
        c_in,
        c_out,
        stride,
        kernel=5,
        expand=2,
        res=True,
        dilation=1,
        se=True,
        **kwargs,
    ),
}


def conv_bn(inp, oup, kernel, stride, pad=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, pad, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class SeparableConv(nn.Module):
    """Separable convolution."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=3, fm_size=7):
        super(SeparableConv, self).__init__()
        assert stride in [1, 2], "stride should be in [1, 2]"
        pad = kernel // 2

        self.conv = nn.Sequential(
            conv_bn(in_ch, in_ch, kernel, stride, pad=pad, groups=in_ch),
            conv_bn(in_ch, out_ch, 1, 1, pad=0),
        )

    def forward(self, x):
        return self.conv(x)


class Conv1x1(nn.Module):
    """1x1 convolution."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=1, fm_size=7):
        super(Conv1x1, self).__init__()
        assert stride in [1, 2], "stride should be in [1, 2]"
        padding = kernel // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):
    """depth convolution."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=3, fm_size=7):
        super(DepthConv, self).__init__()
        assert stride in [1, 2], "stride should be in [1, 2]"
        padding = kernel // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GhostModule(nn.Module):
    """Gost module."""

    def __init__(self, in_ch, out_ch, stride=1, kernel=3, fm_size=7):
        super(GhostModule, self).__init__()
        mid_ch = out_ch // 2
        self.primary_conv = conv_bn(in_ch, mid_ch, 1, stride, pad=0)
        self.cheap_operation = conv_bn(
            mid_ch, mid_ch, kernel, 1, kernel // 2, mid_ch
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, init_ch=32, bottleneck=True):
        super(StemBlock, self).__init__()
        self.stem_1 = conv_bn(in_ch, init_ch, 3, 2, 1)
        mid_ch = int(init_ch // 2) if bottleneck else init_ch
        self.stem_2a = conv_bn(init_ch, mid_ch, 1, 1, 0)
        self.stem_2b = SeparableConv(mid_ch, init_ch, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem_3 = conv_bn(init_ch * 2, init_ch, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out, stem_1_out


class Identity(nn.Module):
    """ Identity module."""

    def __init__(self, in_ch, out_ch, stride=1, fm_size=7):
        super(Identity, self).__init__()
        self.conv = (
            conv_bn(in_ch, out_ch, kernel=1, stride=stride, pad=0)
            if in_ch != out_ch or stride != 1
            else None
        )

    def forward(self, x):
        if self.conv:
            out = self.conv(x)
        else:
            out = x
            # Add dropout to avoid overfit on Identity (PDARTS)
            out = nn.functional.dropout(out, p=0.5)
        return out


class Hsigmoid(nn.Module):
    """Hsigmoid activation function."""

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    """ The improved SE Module."""

    def __init__(self, channel, fm_size=7, se=True):
        super(eSEModule, self).__init__()
        self.se = se

        if self.se:
            self.avg_pool = nn.Conv2d(
                channel, channel, fm_size, 1, 0, groups=channel, bias=False
            )
            self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
            self.hsigmoid = Hsigmoid()

    def forward(self, x):
        if self.se:
            input = x
            x = self.avg_pool(x)
            x = self.fc(x)
            x = self.hsigmoid(x)
            return input * x
        else:
            return x


class ChannelShuffle(nn.Module):
    """Procedure: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]."""

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups == 1:
            return x

        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "group size {} is not for channel {}".format(g, C)
        return (
            x.view(N, g, int(C // g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class MBBlock(nn.Module):
    """The Inverted Residual Block, with channel shuffle or eSEModule."""

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        kernel=3,
        expand=1,
        res=False,
        dilation=1,
        se=False,
        fm_size=7,
        group=1,
        mid_ch=-1,
    ):
        super(MBBlock, self).__init__()
        assert stride in [1, 2], "stride should be in [1, 2]"
        assert kernel in [3, 5], "kernel size should be in [3, 5]"
        assert dilation in [1, 2, 3, 4], "dilation should be in [1, 2, 3, 4]"
        assert group in [1, 2], "group should be in [1, 2]"

        self.use_res_connect = res and (stride == 1)
        padding = kernel // 2 + (dilation - 1)
        mid_ch = mid_ch if mid_ch > 0 else (in_ch * expand)

        # Basic Modules
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.ReLU
        channel_suffle = ChannelShuffle
        se_layer = eSEModule

        self.ir_block = nn.Sequential(
            # pointwise convolution
            conv_layer(in_ch, mid_ch, 1, 1, 0, bias=False, groups=group),
            norm_layer(mid_ch),
            activation_layer(inplace=True),
            # channel shuffle if necessary
            channel_suffle(group),
            # depthwise convolution
            conv_layer(
                mid_ch,
                mid_ch,
                kernel,
                stride,
                padding=padding,
                dilation=dilation,
                groups=mid_ch,
                bias=False,
            ),
            norm_layer(mid_ch),
            # eSEModule if necessary
            se_layer(mid_ch, fm_size, se),
            activation_layer(inplace=True),
            # pointwise convolution
            conv_layer(mid_ch, out_ch, 1, 1, 0, bias=False, groups=group),
            norm_layer(out_ch),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.ir_block(x)
        else:
            return self.ir_block(x)


class SingleOperation(nn.Module):
    """Single operation for sampled path."""

    def __init__(self, layers_configs, stage_ops, sampled_op=""):
        """
        Parameters
        ----------
        layers_configs : list
            the layer config: [input_channel, output_channel, stride, height]
        stage_ops : dict
            the pairs of op name and layer operator
        sampled_op : str
            the searched layer name
        """
        super(SingleOperation, self).__init__()
        fm = {"fm_size": layers_configs[3]}
        ops_names = [op_name for op_name in stage_ops]
        sampled_op = sampled_op if sampled_op else ops_names[0]

        # define the single op
        self.op = stage_ops[sampled_op](*layers_configs[0:3], **fm)

    def forward(self, x):
        return self.op(x)


def choice_blocks(layers_configs, stage_ops):
    """
    Create list of layer candidates for NNI one-shot NAS.

    Parameters
    ----------
    layers_configs : list
        the layer config: [input_channel, output_channel, stride, height]
    stage_ops : dict
        the pairs of op name and layer operator

    Returns
    -------
    output: list
        list of layer operators
    """
    ops_names = [op for op in stage_ops]
    fm = {"fm_size": layers_configs[3]}
    op_list = [stage_ops[op](*layers_configs[0:3], **fm) for op in ops_names]

    return op_list
