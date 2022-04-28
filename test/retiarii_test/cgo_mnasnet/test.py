import os
import sys
import torch
from pathlib import Path

import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.evaluator.pytorch.cgo.evaluator as cgo
from nni.retiarii import serialize
from base_mnasnet import MNASNet
from nni.experiment import RemoteMachineConfig
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.strategy import TPEStrategy
from torchvision import transforms
from torchvision.datasets import CIFAR10

from mutator import BlockMutator

if __name__ == '__main__':
    _DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
    _DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
    _DEFAULT_SKIPS = [False, True, True, True, True, True, True]
    _DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
    _DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]

    base_model = MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                         _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)

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
    train_dataset = serialize(CIFAR10, root='data/cifar10', train=True, download=True, transform=train_transform)
    test_dataset = serialize(CIFAR10, root='data/cifar10', train=False, download=True, transform=valid_transform)
    # trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
    #                             val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
    #                             max_epochs=1, limit_train_batches=0.2)
    trainer = cgo.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=1, limit_train_batches=0.2)

    applied_mutators = [
        BlockMutator('mutable_0'),
        BlockMutator('mutable_1')
    ]

    simple_strategy = TPEStrategy()

    exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_strategy)

    exp_config = RetiariiExeConfig('remote')
    exp_config.experiment_name = 'darts_search'
    exp_config.trial_concurrency = 3
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.reuse_mode = True
    exp_config.max_concurrency_cgo = 3
    exp_config.batch_waiting_time = 0

    rm_conf = RemoteMachineConfig()
    rm_conf.host = '127.0.0.1'
    rm_conf.user = 'xxx'
    rm_conf.password = 'xxx'
    rm_conf.port = 22
    rm_conf.python_path = '/home/xxx/py38/bin'
    rm_conf.gpu_indices = [0, 1, 2]
    rm_conf.use_active_gpu = True
    rm_conf.max_trial_number_per_gpu = 3
    
    exp_config.training_service.machine_list = [rm_conf]
    exp_config.execution_engine = 'cgo'

    exp.run(exp_config, 8099)