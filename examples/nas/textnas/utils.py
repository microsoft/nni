# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
import torch.nn as nn

INF = 1E10
EPS = 1E-12

logger = logging.getLogger("nni.textnas")


def get_length(mask):
    length = torch.sum(mask, 1)
    length = length.long()
    return length


class GlobalAvgPool(nn.Module):
    def forward(self, x, mask):
        x = torch.sum(x, 2)
        length = torch.sum(mask, 1, keepdim=True).float()
        length += torch.eq(length, 0.0).float() * EPS
        length = length.repeat(1, x.size()[1])
        x /= length
        return x


class GlobalMaxPool(nn.Module):
    def forward(self, x, mask):
        mask = torch.eq(mask.float(), 0.0).long()
        mask = torch.unsqueeze(mask, dim=1).repeat(1, x.size()[1], 1)
        mask *= -INF
        x += mask
        x, _ = torch.max(x + mask, 2)
        return x


class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        data = next(self.iterator)
        text, length = data.text
        max_length = text.size(1)
        label = data.label - 1
        bs = label.size(0)
        mask = torch.arange(max_length, device=length.device).unsqueeze(0).repeat(bs, 1)
        mask = mask < length.unsqueeze(-1).repeat(1, max_length)
        return (text, mask), label


def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size
