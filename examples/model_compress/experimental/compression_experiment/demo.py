# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import torch
from torch.optim import Adam

import nni
from nni.compression.experiment.experiment import CompressionExperiment
from nni.compression.experiment.config import CompressionExperimentConfig, TaylorFOWeightPrunerConfig
from vessel import LeNet, finetuner, evaluator, trainer, criterion, device


model = LeNet().to(device)

# pre-training model
finetuner(model)

optimizer = nni.trace(Adam)(model.parameters())

dummy_input = torch.rand(16, 1, 28, 28).to(device)

# normal experiment setting, no need to set search_space and trial_command
config = CompressionExperimentConfig('local')
config.experiment_name = 'auto compression torch example'
config.trial_concurrency = 1
config.max_trial_number = 10
config.trial_code_directory = Path(__file__).parent
config.tuner.name = 'TPE'
config.tuner.class_args['optimize_mode'] = 'maximize'

# compression experiment specific setting
# single float value means the expected remaining ratio upper limit for flops & params, lower limit for metric
config.compression_setting.flops = 0.2
config.compression_setting.params = 0.5
config.compression_setting.module_types = ['Conv2d', 'Linear']
config.compression_setting.exclude_module_names = ['fc2']
config.compression_setting.pruners = [TaylorFOWeightPrunerConfig()]

experiment = CompressionExperiment(config, model, finetuner, evaluator, dummy_input, trainer, optimizer, criterion, device)

experiment.run(8080)
