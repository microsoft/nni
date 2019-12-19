# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from argparse import ArgumentParser
from itertools import cycle

import torch
import torch.nn as nn
from torchtext import data, datasets, vocab

from nni.nas.pytorch.enas import EnasMutator, EnasTrainer
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback

from model import Model


logger = logging.getLogger("nni")


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
    def __init__(self, *args, train_loader=None, valid_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def init_dataloader(self):
        pass


def accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    return (predicted == target).sum().item() / batch_size


def get_data_loader(batch_size, device):
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
    return TEXT.vocab.vectors, cycle(train_loader), cycle(valid_loader), test_loader


if __name__ == "__main__":
    parser = ArgumentParser("textnas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=25, type=int)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embedding, train_loader, valid_loader, test_loader = get_data_loader(args.batch_size, device)
    model = Model(embedding)

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
                             train_loader=train_loader,
                             valid_loader=valid_loader,
                             test_loader=test_loader,
                             log_frequency=args.log_frequency,
                             mutator=mutator,
                             mutator_lr=1E-3,
                             child_steps=3000,
                             mutator_steps=50,
                             mutator_steps_aggregate=10,
                             test_arc_per_epoch=10)
    trainer.train()
