import os
import sys
import torch
from pathlib import Path

from nni.retiarii.trainer import PyTorchImageClassificationTrainer

from base_mnasnet import MNASNet
from nni.retiarii.experiment import RetiariiExperiment, RetiariiExeConfig

from nni.retiarii.strategies import TPEStrategy
from mutator import BlockMutator

if __name__ == '__main__':
    _DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
    _DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
    _DEFAULT_SKIPS = [False, True, True, True, True, True, True]
    _DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
    _DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]

    base_model = MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                    _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)
    trainer = PyTorchImageClassificationTrainer(base_model, dataset_cls="CIFAR10",
            dataset_kwargs={"root": "data/cifar10", "download": True},
            dataloader_kwargs={"batch_size": 32},
            optimizer_kwargs={"lr": 1e-3},
            trainer_kwargs={"max_epochs": 1})

    # new interface
    applied_mutators = []
    applied_mutators.append(BlockMutator('mutable_0'))
    applied_mutators.append(BlockMutator('mutable_1'))

    simple_startegy = TPEStrategy()

    exp = RetiariiExperiment(base_model, trainer, applied_mutators, simple_startegy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'mnasnet_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 10
    exp_config.training_service.use_active_gpu = False

    exp.run(exp_config, 8081, debug=True)
