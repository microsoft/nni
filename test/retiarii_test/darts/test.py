import json
import os
import sys
import torch
from pathlib import Path

from nni.retiarii.experiment import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.strategies import TPEStrategy
from nni.retiarii.trainer import PyTorchImageClassificationTrainer

from darts_model import CNN

if __name__ == '__main__':
    base_model = CNN(32, 3, 16, 10, 8)
    trainer = PyTorchImageClassificationTrainer(base_model, dataset_cls="CIFAR10",
            dataset_kwargs={"root": "data/cifar10", "download": True},
            dataloader_kwargs={"batch_size": 32},
            optimizer_kwargs={"lr": 1e-3},
            trainer_kwargs={"max_epochs": 1})

    simple_startegy = TPEStrategy()

    exp = RetiariiExperiment(base_model, trainer, [], simple_startegy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darts_search'
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True
    exp_config.training_service.gpu_indices = [1, 2]

    exp.run(exp_config, 8081, debug=True)
