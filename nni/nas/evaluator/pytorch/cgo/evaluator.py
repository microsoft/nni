# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Dict, List, Optional, Union, Type


import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader

import nni

from ..lightning import LightningModule, _AccuracyWithLogits, Lightning
from .trainer import Trainer

__all__ = [
    '_MultiModelSupervisedLearningModule', 'MultiModelSupervisedLearningModule',
    '_ClassificationModule', 'Classification',
    '_RegressionModule', 'Regression',
]


@nni.trace
class _MultiModelSupervisedLearningModule(LightningModule):
    def __init__(self, criterion: Type[nn.Module], metrics: Dict[str, Type[torchmetrics.Metric]],
                 n_models: int = 0,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam):
        super().__init__()
        self.save_hyperparameters('criterion', 'optimizer', 'learning_rate', 'weight_decay')
        self.criterion = criterion()
        self.criterion_cls = criterion
        self.optimizer = optimizer
        self.metrics = nn.ModuleDict({name: cls() for name, cls in metrics.items()})
        self.metrics_args = metrics
        self.n_models = n_models

    def dump_kwargs(self):
        kwargs = {}
        kwargs['criterion'] = self.criterion_cls
        kwargs['metrics'] = self.metrics_args
        kwargs['n_models'] = self.n_models
        kwargs['learning_rate'] = self.hparams['learning_rate']
        kwargs['weight_decay'] = self.hparams['weight_decay']
        kwargs['optimizer'] = self.optimizer
        return kwargs

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        multi_y_hat = self(x)
        if isinstance(multi_y_hat, tuple):
            assert len(multi_y_hat) == self.n_models
        else:
            assert self.n_models == 1
            multi_y_hat = [multi_y_hat]
        multi_loss = []
        for idx, y_hat in enumerate(multi_y_hat):
            loss = self.criterion(y_hat.to("cpu"), y.to("cpu"))
            self.log(f'train_loss_{idx}', loss, prog_bar=True)
            for name, metric in self.metrics.items():
                self.log(f'train_{idx}_' + name, metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
            multi_loss.append(loss)
        return sum(multi_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        multi_y_hat = self(x)
        if isinstance(multi_y_hat, tuple):
            assert len(multi_y_hat) == self.n_models
        else:
            assert self.n_models == 1
            multi_y_hat = [multi_y_hat]
        for idx, y_hat in enumerate(multi_y_hat):
            self.log(f'val_loss_{idx}', self.criterion(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
            for name, metric in self.metrics.items():
                self.log(f'val_{idx}_' + name, metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        multi_y_hat = self(x)
        if isinstance(multi_y_hat, tuple):
            assert len(multi_y_hat) == self.n_models
        else:
            assert self.n_models == 1
            multi_y_hat = [multi_y_hat]
        for idx, y_hat in enumerate(multi_y_hat):
            self.log(f'test_loss_{idx}', self.criterion(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
            for name, metric in self.metrics.items():
                self.log(f'test_{idx}_' + name, metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)  # type: ignore

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self._get_validation_metrics())  # type: ignore

    def teardown(self, stage):
        if stage == 'fit':
            nni.report_final_result(self._get_validation_metrics())  # type: ignore

    def _get_validation_metrics(self):
        # TODO: split metric of multiple models?
        if len(self.metrics) == 1:
            metric_name = next(iter(self.metrics))
            ret = []
            for idx in range(self.n_models):
                ret.append(self.trainer.callback_metrics[f'val_{idx}_' + metric_name].item())
            return ret
        else:
            warnings.warn('Multiple metrics without "default" is not supported by current framework.')
            return {name: self.trainer.callback_metrics['val_' + name].item() for name in self.metrics}


class MultiModelSupervisedLearningModule(_MultiModelSupervisedLearningModule):
    """
    Lightning Module of SupervisedLearning for Cross-Graph Optimization.
    Users who needs cross-graph optimization should use this module.

    Parameters
    ----------
    criterion : nn.Module
        Class for criterion module (not an instance). default: ``nn.CrossEntropyLoss``
    learning_rate : float
        Learning rate. default: 0.001
    weight_decay : float
        L2 weight decay. default: 0
    optimizer : Optimizer
        Class for optimizer (not an instance). default: ``Adam``
    """

    def __init__(self, criterion: Type[nn.Module], metrics: Dict[str, Type[torchmetrics.Metric]],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam):
        super().__init__(criterion, metrics, learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)


class _ClassificationModule(_MultiModelSupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam):
        super().__init__(criterion, {'acc': _AccuracyWithLogits},  # type: ignore
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)


class Classification(Lightning):
    """
    Trainer that is used for classification.

    Parameters
    ----------
    criterion : nn.Module
        Class for criterion module (not an instance). default: ``nn.CrossEntropyLoss``
    learning_rate : float
        Learning rate. default: 0.001
    weight_decay : float
        L2 weight decay. default: 0
    optimizer : Optimizer
        Class for optimizer (not an instance). default: ``Adam``
    train_dataloders : DataLoader
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
    val_dataloaders : DataLoader or List of DataLoader
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 **trainer_kwargs):
        module = _ClassificationModule(criterion=criterion, learning_rate=learning_rate,
                                       weight_decay=weight_decay, optimizer=optimizer)
        super().__init__(module, Trainer(use_cgo=True, **trainer_kwargs),
                         train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)


class _RegressionModule(_MultiModelSupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam):
        super().__init__(criterion, {'mse': torchmetrics.MeanSquaredError},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)


class Regression(Lightning):
    """
    Trainer that is used for regression.

    Parameters
    ----------
    criterion : nn.Module
        Class for criterion module (not an instance). default: ``nn.MSELoss``
    learning_rate : float
        Learning rate. default: 0.001
    weight_decay : float
        L2 weight decay. default: 0
    optimizer : Optimizer
        Class for optimizer (not an instance). default: ``Adam``
    train_dataloders : DataLoader
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
    val_dataloaders : DataLoader or List of DataLoader
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 **trainer_kwargs):
        module = _RegressionModule(criterion=criterion, learning_rate=learning_rate,
                                   weight_decay=weight_decay, optimizer=optimizer)
        super().__init__(module, Trainer(use_cgo=True, **trainer_kwargs),
                         train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)
