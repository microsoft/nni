# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import nni
from nni.compression.utils import LightningEvaluator, TorchEvaluator

from .simple_torch_model import training_model, evaluating_model, training_step
from .simple_lightning_model import MNISTDataModule
from ..common import device


def create_lighting_evaluator() -> LightningEvaluator:
    pl_trainer = nni.trace(pl.Trainer)(
        accelerator='auto',
        devices=1,
        max_epochs=1,
        max_steps=50,
        logger=TensorBoardLogger(Path(__file__).parent.parent / 'lightning_logs', name="resnet"),
    )
    pl_trainer.num_sanity_val_steps = 0
    pl_data = nni.trace(MNISTDataModule)(data_dir='data/mnist')
    evaluator = LightningEvaluator(pl_trainer, pl_data, dummy_input=torch.rand(8, 1, 28, 28))
    return evaluator


def create_pytorch_evaluator(model: torch.nn.Module) -> TorchEvaluator:
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = nni.trace(ExponentialLR)(optimizer, 0.1)
    evaluator = TorchEvaluator(training_model, optimizer, training_step, lr_scheduler,
                               dummy_input=torch.rand(8, 1, 28, 28, device=device), evaluating_func=evaluating_model)
    return evaluator
