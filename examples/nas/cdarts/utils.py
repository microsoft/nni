# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import random
from collections import namedtuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from genotypes import Genotype
from ops import PRIMITIVES
from nni.nas.pytorch.cdarts.utils import *


def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger('cdarts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


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


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def parse_results(results, n_nodes):
    concat = range(2, 2 + n_nodes)
    normal_gene = []
    reduction_gene = []
    for i in range(n_nodes):
        normal_node = []
        reduction_node = []
        for j in range(2 + i):
            normal_key = 'normal_n{}_p{}'.format(i + 2, j)
            reduction_key = 'reduce_n{}_p{}'.format(i + 2, j)
            normal_op = results[normal_key].cpu().numpy()
            reduction_op = results[reduction_key].cpu().numpy()
            if sum(normal_op == 1):
                normal_index = np.argmax(normal_op)
                normal_node.append((PRIMITIVES[normal_index], j))
            if sum(reduction_op == 1):
                reduction_index = np.argmax(reduction_op)
                reduction_node.append((PRIMITIVES[reduction_index], j))
        normal_gene.append(normal_node)
        reduction_gene.append(reduction_node)

    genotypes = Genotype(normal=normal_gene, normal_concat=concat,
                         reduce=reduction_gene, reduce_concat=concat)
    return genotypes


def param_size(model, loss_fn, input_size):
    """
    Compute parameter size in MB
    """
    x = torch.rand([2] + input_size).cuda()
    y, _ = model(x)
    target = torch.randint(model.n_classes, size=[2]).cuda()
    loss = loss_fn(y, target)
    loss.backward()
    n_params = sum(np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head') and v.grad is not None)
    return n_params / 1e6


def encode_tensor(data, device):
    if isinstance(data, list):
        if all(map(lambda o: isinstance(o, bool), data)):
            return torch.tensor(data, dtype=torch.bool, device=device)  # pylint: disable=not-callable
        else:
            return torch.tensor(data, dtype=torch.float, device=device)  # pylint: disable=not-callable
    if isinstance(data, dict):
        return {k: encode_tensor(v, device) for k, v in data.items()}
    return data


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
