#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model building blocks factory
"""

import mobile_cv.arch.utils.helper as hp
import mobile_cv.common.misc.registry as registry

from . import basic_blocks as bb
from . import irf_block

PRIMITIVES = registry.Registry("blocks_factory")


_PRIMITIVES = {
    "skip": lambda in_channels, out_channels, stride, **kwargs: bb.Identity(
        in_channels, out_channels, stride
    ),
    "conv": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(conv_args={"stride": stride}, kwargs=kwargs)
    ),
    "conv_k1": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            kwargs=kwargs,
        )
    ),
    "conv_k3": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            kwargs=kwargs,
        )
    ),
    "conv_k5": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            kwargs=kwargs,
        )
    ),
    "conv_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride}, relu_args="hswish", kwargs=kwargs
        )
    ),
    "conv_k1_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 1, "padding": 0},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k3_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 3, "padding": 1},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "conv_k5_hs": lambda in_channels, out_channels, stride, **kwargs: bb.ConvBNRelu(
        in_channels,
        out_channels,
        **hp.merge(
            conv_args={"stride": stride, "kernel_size": 5, "padding": 2},
            relu_args="hswish",
            kwargs=kwargs,
        )
    ),
    "irf": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, **kwargs
    ),
    "ir_k3": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=3, **kwargs
    ),
    "ir_k3_g2": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        pw_groups=2,
        **kwargs
    ),
    "ir_k5": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels, out_channels, stride=stride, kernel_size=5, **kwargs
    ),
    "ir_k5_g2": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        pw_groups=2,
        **kwargs
    ),
    "ir_k3_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k5_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        **kwargs
    ),
    "ir_k3_se": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se",
        **kwargs
    ),
    "ir_k5_se": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se",
        **kwargs
    ),
    "ir_k3_sehsig": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k3_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=3,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_k5_sehsig_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRFBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        kernel_size=5,
        relu_args="hswish",
        se_args="se_hsig",
        **kwargs
    ),
    "ir_pool": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        **hp.filter_kwargs(irf_block.IRPoolBlock, kwargs)
    ),
    "ir_pool_hs": lambda in_channels, out_channels, stride, **kwargs: irf_block.IRPoolBlock(  # noqa
        in_channels,
        out_channels,
        stride=stride,
        relu_args="hswish",
        **hp.filter_kwargs(irf_block.IRPoolBlock, kwargs)
    ),
}
PRIMITIVES.register_dict(_PRIMITIVES)
