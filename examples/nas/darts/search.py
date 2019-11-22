import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LearningRateScheduler)
from nni.nas.pytorch.darts import DartsTrainer
from utils import accuracy

logger = logging.getLogger()

fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
logging.Formatter.converter = time.localtime
formatter = logging.Formatter(fmt, '%m/%d/%Y, %I:%M:%S %p')

std_out_info = logging.StreamHandler()
std_out_info.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(std_out_info)

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")

    model = CNN(32, 3, 16, 10, args.layers)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    trainer = DartsTrainer(model,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                           optimizer=optim,
                           num_epochs=args.epochs,
                           dataset_train=dataset_train,
                           dataset_valid=dataset_valid,
                           batch_size=args.batch_size,
                           log_frequency=args.log_frequency,
                           unrolled=args.unrolled,
                           callbacks=[LearningRateScheduler(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
    trainer.train()
