#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Taken from detectron2

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called "weight" and "bias".
    The two buffers are computed from the original four parameters of BN:
    mean, variance, scale (gamma), offset (beta).
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - mean) / std * scale + offset`, but will be slightly cheaper.
    The pre-trained backbone models from Caffe2 are already in such a frozen format.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return x * scale + bias


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    torch.nn.SyncBatchNorm has bugs. Use this before it is fixed.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
