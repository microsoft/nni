#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing

import torch

from mobile_cv.common.misc.registry import Registry

from .. import lut_ops, lut_schema

PT_CONVERTER = Registry("pytorch_converter")


def get_module_name(m):
    return m.__class__.__name__


@PT_CONVERTER.register("Conv2d")
def convert_Conv2d(m, input_shapes):
    op = lut_ops.Conv2d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.dilation,
        m.groups,
        m.bias is not None,
    )
    ret = lut_schema.OpInfo(op, input_shapes)
    return ret


@PT_CONVERTER.register("ConvTranspose2d")
def convert_ConvTranspose2d(m, input_shapes):
    op = lut_ops.ConvTranspose2d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.output_padding,
        m.groups,
        m.bias is not None,
        m.dilation,
    )
    ret = lut_schema.OpInfo(op, input_shapes)
    return ret


@PT_CONVERTER.register("Linear")
def convert_Linear(m, input_shapes):
    op = lut_ops.Linear(m.in_features, m.out_features, m.bias is not None)
    ret = lut_schema.OpInfo(op, input_shapes)
    return ret


@PT_CONVERTER.register("AdaptiveAvgPool2d")
def convert_AdaptiveAvgPool2d(m, input_shapes):
    op = lut_ops.AdaptiveAvgPool2d(m.output_size)
    ret = lut_schema.OpInfo(op, input_shapes)
    return ret


def convert_module(m: torch.nn.Module, shape):
    name = get_module_name(m)
    func = PT_CONVERTER.get(name, is_raise=False)
    ret = None
    if func is not None:
        ret = func(m, shape)
    return ret


def convert_all_modules(
    model: torch.nn.Module, get_module_shape: typing.Callable
):
    ret = []

    def _convert(m):
        shapes = get_module_shape(m)
        if shapes is not None:
            cur = convert_module(m, shapes)
            if cur is not None:
                ret.append(cur)

    model.apply(_convert)

    return ret
