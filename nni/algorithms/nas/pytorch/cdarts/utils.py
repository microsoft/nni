# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os

import torch
import torch.distributed as dist


class CyclicIterator:
    def __init__(self, loader, sampler, distributed):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self.distributed = distributed
        self._next_epoch()

    def _next_epoch(self):
        if self.distributed:
            self.sampler.set_epoch(self.epoch)
        self.iterator = iter(self.loader)
        self.epoch += 1

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self._next_epoch()
            return next(self.iterator)


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return super().default(o)


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= float(os.environ["WORLD_SIZE"])
    return rt


def reduce_metrics(metrics, distributed=False):
    if distributed:
        return {k: reduce_tensor(v).item() for k, v in metrics.items()}
    return {k: v.item() for k, v in metrics.items()}
