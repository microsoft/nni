# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
from argparse import ArgumentParser
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn

from nni.algorithms.nas.pytorch.enas import EnasMutator, EnasTrainer
from nni.nas.pytorch.callbacks import LRSchedulerCallback

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
    parser.add_argument("--log-frequency", default=50, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, valid_dataset, test_dataset, embedding = read_data_sst("data")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    train_loader, valid_loader = cycle(train_loader), cycle(valid_loader)
    model = Model(embedding)

    mutator = EnasMutator(model, temperature=None, tanh_constant=None, entropy_reduction="mean")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=2e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    trainer = TextNASTrainer(model,
                             loss=criterion,
                             metrics=lambda output, target: {"acc": accuracy(output, target)},
                             reward_function=accuracy,
                             optimizer=optimizer,
                             callbacks=[LRSchedulerCallback(lr_scheduler)],
                             batch_size=args.batch_size,
                             num_epochs=args.epochs,
                             dataset_train=None,
                             dataset_valid=None,
                             train_loader=train_loader,
                             valid_loader=valid_loader,
                             test_loader=test_loader,
                             log_frequency=args.log_frequency,
                             mutator=mutator,
                             mutator_lr=2e-3,
                             mutator_steps=500,
                             mutator_steps_aggregate=1,
                             child_steps=3000,
                             baseline_decay=0.99,
                             test_arc_per_epoch=10)
    trainer.train()
    os.makedirs("checkpoints", exist_ok=True)
    for i in range(20):
        trainer.export(os.path.join("checkpoints", "architecture_%02d.json" % i))
