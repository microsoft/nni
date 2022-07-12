# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json

import nni
import numpy as np
import torch

from nni.retiarii import strategy, fixed_arch
from nni.retiarii.evaluator.pytorch import Lightning, ClassificationModule, Trainer
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch import NasBench201
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing_extensions import Literal


@nni.trace
class TimmTrainingModule(ClassificationModule):
    """Implementation of several features in https://github.com/rwightman/pytorch-image-models/blob/master/train.py
    with PyTorch-Lightning."""

    def __init__(self,
                 learning_rate: float = 0.1,
                 weight_decay: float = 5e-4,
                 max_epochs: int = 200):
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)



    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.weight_decay,  # type: ignore
            nesterov=True,
        )
        return {
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=0)
        }
