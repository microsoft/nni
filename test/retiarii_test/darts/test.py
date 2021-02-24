import json
import os
import sys
import torch
from pathlib import Path

import nni.retiarii.trainer.pytorch.lightning as pl
import nni.retiarii.strategy as strategy
from nni.retiarii import basic_unit as bm
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from torchvision import transforms
from torchvision.datasets import CIFAR10

from darts_model import CNN

if __name__ == '__main__':
    base_model = CNN(32, 3, 16, 10, 8)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = bm(CIFAR10)(root='data/cifar10', train=True, download=True, transform=train_transform)
    test_dataset = bm(CIFAR10)(root='data/cifar10', train=False, download=True, transform=valid_transform)
    trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=1, limit_train_batches=0.2)

    simple_strategy = strategy.Random()

    exp = RetiariiExperiment(base_model, trainer, [], simple_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darts_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True
    exp_config.training_service.gpu_indices = [1, 2]

    exp.run(exp_config, 8081)
