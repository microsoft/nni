#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model inverse residual building block
"""

import numbers

import torch.nn as nn

import mobile_cv.arch.utils.helper as hp
import mobile_cv.common.misc.registry as registry

from . import basic_blocks as bb

RESIDUAL_REGISTRY = registry.Registry("residual_connect")


def build_residual_connect(
    name, in_channels, out_channels, stride, drop_connect_rate=None, **res_args
):
    if name is None:
        return None
    if name == "default":
        assert isinstance(stride, (numbers.Number, tuple, list))
        if isinstance(stride, (tuple, list)):
            stride_one = all(x == 1 for x in stride)
        else:
            stride_one = stride == 1
        if in_channels == out_channels and stride_one:
            if drop_connect_rate is None:
                return bb.TorchAdd()
            else:
                return bb.AddWithDropConnect(drop_connect_rate)
        else:
            return None
    return RESIDUAL_REGISTRY.get(name)(
        in_channels, out_channels, stride, **res_args
    )


class IRFBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=True,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        res_conn_args="default",
        upsample_args="default",
        width_divisor=8,
        pw_args=None,
        dw_args=None,
        pwl_args=None,
        dw_skip_bnrelu=False,
        pw_groups=1,
        always_pw=False,
        less_se_channels=False,
        zero_last_bn_gamma=True,
        drop_connect_rate=None,
    ):
        super().__init__()

        conv_args = hp.unify_args(conv_args)
        bn_args = hp.unify_args(bn_args)
        relu_args = hp.unify_args(relu_args)

        mid_channels = hp.get_divisible_by(
            in_channels * expansion, width_divisor
        )

        res_conn = build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate,
            **hp.unify_args(res_conn_args)
        )

        self.pw = None
        self.shuffle = None
        if in_channels != mid_channels or always_pw:
            self.pw = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    "groups": pw_groups,
                    **hp.merge_unify_args(conv_args, pw_args),
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )
        if pw_groups > 1:
            self.shuffle = bb.ChannelShuffle(pw_groups)
        # use negative stride for upsampling
        self.upsample, dw_stride = bb.build_upsample_neg_stride(
            stride=stride, **hp.unify_args(upsample_args)
        )
        self.dw = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=mid_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": dw_stride,
                "padding": kernel_size // 2,
                "groups": mid_channels,
                "bias": bias,
                **hp.merge_unify_args(conv_args, dw_args),
            },
            bn_args=bn_args if not dw_skip_bnrelu else None,
            relu_args=relu_args if not dw_skip_bnrelu else None,
        )
        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=int(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **hp.merge(relu_args=relu_args, kwargs=hp.unify_args(se_args))
        )
        self.pwl = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                "groups": pw_groups,
                **hp.merge_unify_args(conv_args, pwl_args),
            },
            bn_args={
                **bn_args,
                **{
                    "zero_gamma": (
                        zero_last_bn_gamma if res_conn is not None else False
                    )
                },
            },
            relu_args=None,
        )

        self.res_conn = res_conn
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw:
            y = self.pw(y)
        if self.shuffle:
            y = self.shuffle(y)
        if self.upsample:
            y = self.upsample(y)
        if self.dw:
            y = self.dw(y)
        if self.se:
            y = self.se(y)
        if self.pwl:
            y = self.pwl(y)
        if self.res_conn:
            y = self.res_conn(y, x)
        return y


class IRPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=-1,
        stride=1,
        bias=True,
        pw_args="conv",
        pwl_args="conv",
        pool_args=None,
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        res_conn_args="default",
        width_divisor=8,
        always_pw=False,
        less_se_channels=False,
    ):
        super().__init__()

        mid_channels = hp.get_divisible_by(
            in_channels * expansion, width_divisor
        )

        self.pw = None
        if in_channels != mid_channels or always_pw:
            self.pw = bb.ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    **hp.unify_args(pw_args),
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )

        if kernel_size == -1:
            self.dw = nn.AdaptiveAvgPool2d(1)
        else:
            self.dw = nn.AvgPool2d(
                kernel_size, stride=stride, **hp.unify_args(pool_args)
            )

        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion
        self.se = bb.build_se(
            in_channels=mid_channels,
            mid_channels=(mid_channels * se_ratio),
            width_divisor=width_divisor,
            relu_args=relu_args,
            **hp.unify_args(se_args)
        )
        self.pwl = bb.ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                **hp.merge_unify_args(pwl_args),
            },
            # no bn
            bn_args=None,
            # has relu
            relu_args=relu_args,
        )
        self.res_conn = build_residual_connect(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            **hp.unify_args(res_conn_args)
        )
        self.out_channels = out_channels

    def forward(self, x):
        y = x
        if self.pw:
            y = self.pw(y)
        if self.dw:
            y = self.dw(y)
        if self.se:
            y = self.se(y)
        if self.pwl:
            y = self.pwl(y)
        if self.res_conn:
            y = self.res_conn(y, x)
        return y
