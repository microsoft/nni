# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import warnings
from pathlib import Path
from typing import Dict, NoReturn, Union, Optional, List, Type

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader

import nni
try:
    from .cgo import trainer as cgo_trainer
    cgo_import_failed = False
except ImportError:
    cgo_import_failed = True

from ...graph import Evaluator
from ...serializer import serialize_cls


__all__ = ['LightningModule', 'Trainer', 'DataLoader', 'Lightning', 'Classification', 'Regression']


class LightningModule(pl.LightningModule):
    """
    Basic wrapper of generated model.

    Lightning modules used in NNI should inherit this class.
    """

    def set_model(self, model: Union[Type[nn.Module], nn.Module]) -> NoReturn:
        if isinstance(model, type):
            self.model = model()
        else:
            self.model = model


Trainer = serialize_cls(pl.Trainer)
DataLoader = serialize_cls(DataLoader)

class Lightning(Evaluator):
    """
    Delegate the whole training to PyTorch Lightning.

    Since the arguments passed to the initialization needs to be serialized, ``LightningModule``, ``Trainer`` or
    ``DataLoader`` in this file should be used. Another option is to hide dataloader in the Lightning module, in
    which case, dataloaders are not required for this class to work.

    Following the programming style of Lightning, metrics sent to NNI should be obtained from ``callback_metrics``
    in trainer. Two hooks are added at the end of validation epoch and the end of ``fit``, respectively. The metric name
    and type depend on the specific task.

    Parameters
    ----------
    lightning_module : LightningModule
        Lightning module that defines the training logic.
    trainer : Trainer
        Lightning trainer that handles the training.
    train_dataloders : DataLoader
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
    val_dataloaders : DataLoader or List of DataLoader
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
    """

    def __init__(self, lightning_module: LightningModule, trainer: Trainer,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None):
        assert isinstance(lightning_module, LightningModule), f'Lightning module must be an instance of {__name__}.LightningModule.'
        if cgo_import_failed:
            assert isinstance(trainer, Trainer), f'Trainer must be imported from {__name__}'
        else:
            assert isinstance(trainer, Trainer) or isinstance(trainer, cgo_trainer.Trainer), \
                f'Trainer must be imported from {__name__} or nni.retiarii.evaluator.pytorch.cgo.trainer'
        assert _check_dataloader(train_dataloader), f'Wrong dataloader type. Try import DataLoader from {__name__}.'
        assert _check_dataloader(val_dataloaders), f'Wrong dataloader type. Try import DataLoader from {__name__}.'
        self.module = lightning_module
        self.trainer = trainer
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders

    @staticmethod
    def _load(ir):
        return Lightning(ir['module'], ir['trainer'], ir['train_dataloader'], ir['val_dataloaders'])

    def _dump(self):
        return {
            'module': self.module,
            'trainer': self.trainer,
            'train_dataloader': self.train_dataloader,
            'val_dataloaders': self.val_dataloaders
        }

    def _execute(self, model_cls):
        return self.fit(model_cls)

    def __eq__(self, other):
        eq_func = False
        eq_args = False
        if other is None:
            return False
        if hasattr(self, "function") and hasattr(other, "function"):
            eq_func = (self.function == other.function)
        elif not (hasattr(self, "function") or hasattr(other, "function")):
            eq_func = True

        if hasattr(self, "arguments") and hasattr(other, "arguments"):
            eq_args = (self.arguments == other.arguments)
        elif not (hasattr(self, "arguments") or hasattr(other, "arguments")):
            eq_args = True

        return eq_func and eq_args

    def fit(self, model):
        """
        Fit the model with provided dataloader, with Lightning trainer.

        Parameters
        ----------
        model : nn.Module
            The model to fit.
        """
        self.module.set_model(model)
        return self.trainer.fit(self.module, self.train_dataloader, self.val_dataloaders)


def _check_dataloader(dataloader):
    if dataloader is None:
        return True
    if isinstance(dataloader, list):
        return all([_check_dataloader(d) for d in dataloader])
    return isinstance(dataloader, DataLoader)


### The following are some commonly used Lightning modules ###

class _SupervisedLearningModule(LightningModule):
    def __init__(self, criterion: nn.Module, metrics: Dict[str, torchmetrics.Metric],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam,
                 export_onnx: Union[Path, str, bool, None] = None):
        super().__init__()
        self.save_hyperparameters('criterion', 'optimizer', 'learning_rate', 'weight_decay')
        self.criterion = criterion()
        self.optimizer = optimizer
        self.metrics = nn.ModuleDict({name: cls() for name, cls in metrics.items()})

        if export_onnx is None or export_onnx is True:
            self.export_onnx = Path(os.environ.get('NNI_OUTPUT_DIR', '.')) / 'model.onnx'
            self.export_onnx.parent.mkdir(exist_ok=True)
        elif export_onnx:
            self.export_onnx = Path(export_onnx)
        else:
            self.export_onnx = None
        self._already_exported = False

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if not self._already_exported:
            try:
                self.to_onnx(self.export_onnx, x, export_params=True)
            except RuntimeError as e:
                warnings.warn(f'ONNX conversion failed. As a result, you might not be able to use visualization. Error message: {e}')
            self._already_exported = True

        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('val_' + name, metric(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('test_loss', self.criterion(y_hat, y), prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('test_' + name, metric(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self._get_validation_metrics())

    def on_fit_end(self):
        nni.report_final_result(self._get_validation_metrics())

    def _get_validation_metrics(self):
        if len(self.metrics) == 1:
            metric_name = next(iter(self.metrics))
            return self.trainer.callback_metrics['val_' + metric_name].item()
        else:
            warnings.warn('Multiple metrics without "default" is not supported by current framework.')
            return {name: self.trainer.callback_metrics['val_' + name].item() for name in self.metrics}


class _AccuracyWithLogits(torchmetrics.Accuracy):
    def update(self, pred, target):
        return super().update(nn.functional.softmax(pred), target)


@serialize_cls
class _ClassificationModule(_SupervisedLearningModule):
    def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam,
                 export_onnx: bool = True):
        super().__init__(criterion, {'acc': _AccuracyWithLogits},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


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
    export_onnx : bool
        If true, model will be exported to ``model.onnx`` before training starts. default true
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 export_onnx: bool = True,
                 **trainer_kwargs):
        module = _ClassificationModule(criterion=criterion, learning_rate=learning_rate,
                                       weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)


@serialize_cls
class _RegressionModule(_SupervisedLearningModule):
    def __init__(self, criterion: nn.Module = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam,
                 export_onnx: bool = True):
        super().__init__(criterion, {'mse': torchmetrics.MeanSquaredError},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


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
    export_onnx : bool
        If true, model will be exported to ``model.onnx`` before training starts. default: true
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, criterion: nn.Module = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 export_onnx: bool = True,
                 **trainer_kwargs):
        module = _RegressionModule(criterion=criterion, learning_rate=learning_rate,
                                   weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)
