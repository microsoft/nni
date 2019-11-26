# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

_counter = 0


def global_mutable_counting():
    global _counter
    _counter += 1
    return _counter


class AverageMeterGroup:

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __str__(self):
        return "  ".join(str(v) for _, v in self.meters.items())


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter

        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class StructuredMutableTreeNode:
    """
    A structured representation of a search space.
    A search space comes with a root (with `None` stored in its `mutable`), and a bunch of children in its `children`.
    This tree can be seen as a "flattened" version of the module tree. Since nested mutable entity is not supported yet,
    the following must be true: each subtree corresponds to a ``MutableScope`` and each leaf corresponds to a
    ``Mutable`` (other than ``MutableScope``).
    """

    def __init__(self, mutable):
        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):
        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):
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
