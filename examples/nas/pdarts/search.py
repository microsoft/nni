# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

from nni.nas.pytorch.callbacks import ArchitectureCheckpoint
from nni.nas.pytorch.pdarts import PdartsTrainer

# prevent it to be reordered.
if True:
    sys.path.append('../darts')
    from utils import accuracy
    from model import CNN
    import datasets

logger = logging.getLogger()

fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
logging.Formatter.converter = time.localtime
formatter = logging.Formatter(fmt, '%m/%d/%Y, %I:%M:%S %p')

std_out_info = logging.StreamHandler()
std_out_info.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(std_out_info)

if __name__ == "__main__":
    parser = ArgumentParser("pdarts")
    parser.add_argument('--add_layers', action='append',
                        default=[0, 6, 12], help='add layers')
    parser.add_argument("--nodes", default=4, type=int)
    parser.add_argument("--layers", default=5, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    logger.info("loading data")
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    def model_creator(layers):
        model = CNN(32, 3, 16, 10, layers, n_nodes=args.nodes)
        criterion = nn.CrossEntropyLoss()

        optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

        return model, criterion, optim, lr_scheduler

    logger.info("initializing trainer")
    trainer = PdartsTrainer(model_creator,
                            layers=args.layers,
                            metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                            pdarts_num_layers=[0, 6, 12],
                            pdarts_num_to_drop=[3, 2, 2],
                            num_epochs=args.epochs,
                            dataset_train=dataset_train,
                            dataset_valid=dataset_valid,
                            batch_size=args.batch_size,
                            log_frequency=args.log_frequency,
                            callbacks=[ArchitectureCheckpoint("./checkpoints")])
    logger.info("training")
    trainer.train()
