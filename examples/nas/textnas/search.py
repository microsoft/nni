# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from argparse import ArgumentParser
from itertools import cycle

import torch
import torch.nn as nn

from nni.nas.pytorch.enas import EnasMutator, EnasTrainer
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback

from dataloader import read_data_sst
from model import Model
from utils import accuracy


logger = logging.getLogger("nni.textnas")


class TextNASTrainer(EnasTrainer):
    def __init__(self, *args, train_loader=None, valid_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def init_dataloader(self):
        pass


if __name__ == "__main__":
    parser = ArgumentParser("textnas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=25, type=int)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, valid_dataset, test_dataset, embedding = read_data_sst("data")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    train_loader, valid_loader = cycle(train_loader), cycle(valid_loader)
    model = Model(embedding)

    num_epochs = 10
    mutator = EnasMutator(model, tanh_constant=None, entropy_reduction="mean")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, eps=1E-3, weight_decay=2E-6)
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
                             mutator_lr=2E-3,
                             mutator_steps=500,
                             mutator_steps_aggregate=1,
                             child_steps=3000,
                             skip_weight=0.,
                             entropy_weight=0.,
                             baseline_decay=0.99,
                             test_arc_per_epoch=10)
    trainer.train()
