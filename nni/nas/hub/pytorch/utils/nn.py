# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Miscellaneous neural network utilities."""

from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is copied from ``timm.layers.DropPath`` and modified to support both NASNet and AutoFormer,
    as well as the shape inference.

    Credit partially goes to `pt.darts <https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py>`__.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Drops some samples in the forward pass with probability ``drop_prob``.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def _shape_forward(self, x):
        return x.real_shape

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
