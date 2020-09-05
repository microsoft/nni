#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from detectron2

"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math

import torch
from torch.nn.modules.utils import _ntuple, _pair


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single
    element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def _get_conv_2d_output_shape(conv_args, x):
    # When input is empty, we want to return a empty tensor with "correct" shape,
    # So that the following operations will not panic
    # if they check for the shape of the tensor.
    # This computes the height and width of the output tensor
    output_shape = [
        (i + 2 * p - (di * (k - 1) + 1)) // s + 1
        for i, p, di, k, s in zip(
            x.shape[-2:],
            conv_args.padding,
            conv_args.dilation,
            conv_args.kernel_size,
            conv_args.stride,
        )
    ]
    output_shape = [x.shape[0], conv_args.out_channels] + output_shape
    return output_shape


class Conv2dEmptyOutput(torch.nn.Module):
    def __init__(self, conv_op):
        super().__init__()
        assert isinstance(conv_op, torch.nn.Conv2d)
        self.padding = conv_op.padding
        self.dilation = conv_op.dilation
        self.kernel_size = conv_op.kernel_size
        self.stride = conv_op.stride
        self.out_channels = conv_op.out_channels

    def forward(self, x):
        assert x.numel() == 0, "Only handle empty batch"
        output_shape = _get_conv_2d_output_shape(self, x)
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        norm (nn.Module, optional): a normalization layer
        activation (callable(Tensor) -> Tensor): a callable activation function
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0:
            output_shape = _get_conv_2d_output_shape(self, x)
            return _NewEmptyTensorOp.apply(x, output_shape)

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class AvgPool2d(torch.nn.AvgPool2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(AvgPool2d, self).forward(x)
        # get output shape
        floor_func = math.floor if not self.ceil_mode else math.ceil
        output_shape = [
            int(floor_func((i + 2 * p - k) / s + 1))
            for i, p, k, s in zip(
                x.shape[-2:],
                _pair(self.padding),
                _pair(self.kernel_size),
                _pair(self.stride),
            )
        ]
        output_shape = [x.shape[0], x.shape[1]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class GroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        if x.numel() > 0:
            return super(GroupNorm, self).forward(x)

        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError(
                "only one of size or scale_factor should be defined"
            )
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(
                    dim, len(scale_factor)
                )
            )

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i]))
            for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
