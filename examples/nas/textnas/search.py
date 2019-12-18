# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchtext import data, datasets, vocab

from nni.nas.pytorch.enas import EnasMutator, EnasTrainer
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback

from model import Model


logger = logging.getLogger("nni")
vocab_glove = vocab.GloVe(cache="data")


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


class TextNASTrainer(EnasTrainer):
    def init_dataloader(self):
        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = data.Field(sequential=False)
        train, val, test = datasets.SST.splits(TEXT, LABEL, root="data")
        TEXT.build_vocab(train, vectors=vocab_glove)
        LABEL.build_vocab(train)
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=128, device=self.device)
        self.train_loader = IteratorWrapper(train_iter)
        self.valid_loader = IteratorWrapper(val_iter)
        self.test_loader = IteratorWrapper(test_iter)
        logger.info("Loaded %d batches for training, %d for validation, %d for testing.",
                    len(self.train_loader), len(self.valid_loader), len(self.test_loader))


def accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


if __name__ == "__main__":
    parser = ArgumentParser("textnas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    args = parser.parse_args()

    model = Model(vocab_glove.vectors)

    num_epochs = 10
    mutator = EnasMutator(model, tanh_constant=None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, eps=1E-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    trainer = TextNASTrainer(model,
                             loss=criterion,
                             metrics=lambda output, target: {"acc": accuracy(output, target)},
                             reward_function=accuracy,
                             optimizer=optimizer,
                             callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                             batch_size=args.batch_size,
                             num_epochs=num_epochs,
                             dataset_train=None,
                             dataset_valid=None,
                             log_frequency=args.log_frequency,
                             mutator=mutator,
                             mutator_lr=1E-3)
    import time
    input()
    trainer.train()
