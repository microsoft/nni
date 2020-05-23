# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf

_counter = 0

def global_mutable_counting():
    global _counter
    _counter += 1
    return _counter


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def __str__(self):
        return '{name} {val:4f} ({avg:4f})'.format(**self.__dict__)

    def summary(self):
        return '{name}: {avg:4f}'.format(**self.__dict__)


class AverageMeterGroup:
    def __init__(self):
        self.meters = {}

    def update(self, data):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k)
            self.meters[k].update(v)

    def __str__(self):
        return '  '.join(str(v) for v in self.meters.values())

    def summary(self):
        return '  '.join(v.summary() for v in self.meters.values())


class StructuredMutableTreeNode:
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


def fill_zero_grads(grads, weights):
    ret = []
    for grad, weight in zip(grads, weights):
        if grad is not None:
            ret.append(grad)
        else:
            ret.append(tf.zeros_like(weight))
    return ret
