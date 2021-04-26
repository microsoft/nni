# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

from nni.nas.pytorch.callbacks import ArchitectureCheckpoint
from nni.algorithms.nas.pytorch.pdarts import PdartsTrainer

# prevent it to be reordered.
if True:
    sys.path.append('../darts')
    from utils import accuracy
    from model import CNN
    import datasets


logger = logging.getLogger('nni')


if __name__ == "__main__":
    parser = ArgumentParser("pdarts")
    parser.add_argument('--add_layers', action='append', type=int,
                        help='add layers, default: [0, 6, 12]')
    parser.add_argument('--dropped_ops', action='append', type=int,
                        help='drop ops, default: [3, 2, 1]')
    parser.add_argument("--nodes", default=4, type=int)
    parser.add_argument("--init_layers", default=5, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    args = parser.parse_args()
    if args.add_layers is None:
        args.add_layers = [0, 6, 12]
    if args.dropped_ops is None:
        args.dropped_ops = [3, 2, 1]

    logger.info("loading data")
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    def model_creator(layers):
        model = CNN(32, 3, args.channels, 10, layers, n_nodes=args.nodes)
        criterion = nn.CrossEntropyLoss()

        optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

        return model, criterion, optim, lr_scheduler

    logger.info("initializing trainer")
    trainer = PdartsTrainer(model_creator,
                            init_layers=args.init_layers,
                            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                            pdarts_num_layers=args.add_layers,
                            pdarts_num_to_drop=args.dropped_ops,
                            num_epochs=args.epochs,
                            dataset_train=dataset_train,
                            dataset_valid=dataset_valid,
                            batch_size=args.batch_size,
                            log_frequency=args.log_frequency,
                            unrolled=args.unrolled,
                            callbacks=[ArchitectureCheckpoint("./checkpoints")])
    logger.info("training")
    trainer.train()
