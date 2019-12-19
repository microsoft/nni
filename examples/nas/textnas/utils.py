import logging
import os
from itertools import cycle

import torch
import torch.nn as nn
from torchtext import data, datasets, vocab

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
        label = data.label
        bs = label.size(0)
        mask = torch.arange(max_length, device=length.device).unsqueeze(0).repeat(bs, 1)
        mask = mask < length.unsqueeze(-1).repeat(1, max_length)
        return (text, mask), label


def accuracy(output, target):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


def get_data_loader(batch_size, device, infinite=True):
    data_folder = "data"
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    sst_folder = datasets.SST.download(data_folder)
    train = datasets.SST(os.path.join(sst_folder, "train.txt"), TEXT, LABEL, subtrees=True)
    val = datasets.SST(os.path.join(sst_folder, "dev.txt"), TEXT, LABEL, subtrees=True)
    test = datasets.SST(os.path.join(sst_folder, "test.txt"), TEXT, LABEL)
    TEXT.build_vocab(train, vectors=vocab.GloVe(cache=data_folder))
    LABEL.build_vocab(train)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=device)
    train_loader = IteratorWrapper(train_iter)
    valid_loader = IteratorWrapper(val_iter)
    test_loader = IteratorWrapper(test_iter)
    logger.info("Loaded %d batches for training, %d for validation, %d for testing.",
                len(train_loader), len(valid_loader), len(test_loader))
    if infinite:
        train_loader = cycle(train_loader)
        valid_loader = cycle(valid_loader)
    return TEXT.vocab.vectors, train_loader, valid_loader, test_loader
