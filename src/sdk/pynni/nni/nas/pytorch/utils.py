# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from collections import OrderedDict

import numpy as np
import torch

_counter = 0

_logger = logging.getLogger(__name__)


def global_mutable_counting():
    """
    A program level counter starting from 1.
    """
    global _counter
    _counter += 1
    return _counter


def _reset_global_mutable_counting():
    """
    Reset the global mutable counting to count from 1. Useful when defining multiple models with default keys.
    """
    global _counter
    _counter = 0


def to_device(obj, device):
    """
    Move a tensor, tuple, list, or dict onto device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def to_list(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy().tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return arr


class AverageMeterGroup:
    """
    Average meter group for multiple average meters.
    """

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        """
        Update the meter group with a dict of metrics.
        Non-exist average meters will be automatically created.
        """
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        """
        Return a summary string of group data.
        """
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class StructuredMutableTreeNode:
    """
    A structured representation of a search space.
    A search space comes with a root (with `None` stored in its `mutable`), and a bunch of children in its `children`.
    This tree can be seen as a "flattened" version of the module tree. Since nested mutable entity is not supported yet,
    the following must be true: each subtree corresponds to a ``MutableScope`` and each leaf corresponds to a
    ``Mutable`` (other than ``MutableScope``).

    Parameters
    ----------
    mutable : nni.nas.pytorch.mutables.Mutable
        The mutable that current node is linked with.
    """

    def __init__(self, mutable):
        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):
        """
        Add a tree node to the children list of current node.
        """
        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):
        """
        Return the ``type`` of mutable content.
        """
        return type(self.mutable)

    def __iter__(self):
        return self.traverse()

    def traverse(self, order="pre", deduplicate=True, memo=None):
        """
        Return a generator that generates a list of mutables in this tree.

        Parameters
        ----------
        order : str
            pre or post. If pre, current mutable is yield before children. Otherwise after.
        deduplicate : bool
            If true, mutables with the same key will not appear after the first appearance.
        memo : dict
            An auxiliary dict that memorize keys seen before, so that deduplication is possible.

        Returns
        -------
        generator of Mutable
        """
        if memo is None:
            memo = set()
        assert order in ["pre", "post"]
        if order == "pre":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
        for child in self.children:
            for m in child.traverse(order=order, deduplicate=deduplicate, memo=memo):
                yield m
        if order == "post":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
