# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Union, Optional, List, Callable, Type

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torchmetrics
import torch.utils.data as torch_data

import nni
from nni.common.serializer import is_traceable
try:
    from .cgo import trainer as cgo_trainer
    cgo_import_failed = False
except ImportError:
    cgo_import_failed = True

from nni.retiarii.graph import Evaluator
from nni.typehint import Literal


__all__ = ['LightningModule', 'Trainer', 'DataLoader', 'Lightning', 'Classification', 'Regression']


class LightningModule(pl.LightningModule):
    """
    Basic wrapper of generated model.
    Lightning modules used in NNI should inherit this class.

    It's a subclass of ``pytorch_lightning.LightningModule``.
    See https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    running_mode: Literal['multi', 'oneshot'] = 'multi'
    """An indicator of whether current module is running in a multi-trial experiment or an one-shot.
    This flag should be automatically set by experiments when they start to run.
    """

    def set_model(self, model: Union[Callable[[], nn.Module], nn.Module]) -> None:
        """Set the inner model (architecture) to train / evaluate.

        Parameters
        ----------
        model : callable or nn.Module
            Can be a callable returning nn.Module or nn.Module.
        """
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = model()


Trainer = nni.trace(pl.Trainer)
Trainer.__doc__ = """
Traced version of ``pytorch_lightning.Trainer``. See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
"""
DataLoader = nni.trace(torch_data.DataLoader)
DataLoader.__doc__ = """
Traced version of ``torch.utils.data.DataLoader``. See https://pytorch.org/docs/stable/data.html
"""


@nni.trace
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
    lightning_module
        Lightning module that defines the training logic.
    trainer
        Lightning trainer that handles the training.
    train_dataloders
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
        It can be `any types of dataloader supported by Lightning <https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html>`__.
    val_dataloaders
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
        It can be `any types of dataloader supported by Lightning <https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html>`__.
    """

    def __init__(self, lightning_module: LightningModule, trainer: Trainer,
                 train_dataloaders: Optional[Any] = None,
                 val_dataloaders: Optional[Any] = None,
                 train_dataloader: Optional[Any] = None):
        assert isinstance(lightning_module, LightningModule), f'Lightning module must be an instance of {__name__}.LightningModule.'
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        if cgo_import_failed:
            assert isinstance(trainer, pl.Trainer) and is_traceable(trainer), f'Trainer must be imported from {__name__}'
        else:
            # this is not isinstance(trainer, Trainer) because with a different trace call, it can be different
            assert (isinstance(trainer, pl.Trainer) and is_traceable(trainer)) or isinstance(trainer, cgo_trainer.Trainer), \
                f'Trainer must be imported from {__name__} or nni.retiarii.evaluator.pytorch.cgo.trainer'
        if not _check_dataloader(train_dataloaders):
            warnings.warn(f'Please try to wrap PyTorch DataLoader with nni.trace or '
                          f'import DataLoader from {__name__}: {train_dataloaders}',
                          RuntimeWarning)
        if not _check_dataloader(val_dataloaders):
            warnings.warn(f'Please try to wrap PyTorch DataLoader with nni.trace or '
                          f'import DataLoader from {__name__}: {val_dataloaders}',
                          RuntimeWarning)
        self.module = lightning_module
        self.trainer = trainer
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders

    @staticmethod
    def _load(ir):
        return Lightning(ir['module'], ir['trainer'], ir['train_dataloaders'], ir['val_dataloaders'])

    def _dump(self):
        return {
            'type': self.__class__,
            'module': self.module,
            'trainer': self.trainer,
            'train_dataloaders': self.train_dataloaders,
            'val_dataloaders': self.val_dataloaders
        }

    def _execute(self, model_cls):
        return self.fit(model_cls)

    @property
    def train_dataloader(self):
        warnings.warn('train_dataloader is deprecated, please use `train_dataloaders`.', DeprecationWarning)

    def __eq__(self, other):
        eq_func = False
        eq_args = False
        if other is None:
            return False
        if hasattr(self, "function") and hasattr(other, "function"):
            eq_func = getattr(self, "function") == getattr(other, "function")
        elif not (hasattr(self, "function") or hasattr(other, "function")):
            eq_func = True

        if hasattr(self, "arguments") and hasattr(other, "arguments"):
            eq_args = getattr(self, "arguments") == getattr(other, "arguments")
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
        return self.trainer.fit(self.module, self.train_dataloaders, self.val_dataloaders)


def _check_dataloader(dataloader):
    # Check the type of dataloader recursively.
    if isinstance(dataloader, list):
        return all([_check_dataloader(d) for d in dataloader])
    if isinstance(dataloader, dict):
        return all([_check_dataloader(v) for v in dataloader.values()])
    if isinstance(dataloader, torch_data.DataLoader):
        return is_traceable(dataloader)
    return True


### The following are some commonly used Lightning modules ###

class _SupervisedLearningModule(LightningModule):

    trainer: pl.Trainer

    def __init__(self, criterion: Type[nn.Module], metrics: Dict[str, Type[torchmetrics.Metric]],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: Union[Path, str, bool, None] = None):
        super().__init__()
        self.save_hyperparameters('criterion', 'optimizer', 'learning_rate', 'weight_decay')
        self.criterion = criterion()
        self.optimizer = optimizer
        self.metrics = nn.ModuleDict({name: cls() for name, cls in metrics.items()})

        if export_onnx is None or export_onnx is True:
            self.export_onnx = Path(os.environ.get('NNI_OUTPUT_DIR', '.')) / 'model.onnx'
        elif export_onnx:
            self.export_onnx = Path(export_onnx)
        else:
            self.export_onnx = None

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

        if self.running_mode == 'multi' and self.export_onnx is not None:
            self.export_onnx.parent.mkdir(exist_ok=True)
            try:
                self.to_onnx(self.export_onnx, x, export_params=True)
            except RuntimeError as e:
                warnings.warn(f'ONNX conversion failed. As a result, you might not be able to use visualization. Error message: {e}')
            self.export_onnx = None

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
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)  # type: ignore

    def on_validation_epoch_end(self):
        if self.running_mode == 'multi':
            nni.report_intermediate_result(self._get_validation_metrics())

    def on_fit_end(self):
        if self.running_mode == 'multi':
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
        return super().update(nn_functional.softmax(pred, dim=-1), target)


@nni.trace
class _ClassificationModule(_SupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: bool = True):
        super().__init__(criterion, {'acc': _AccuracyWithLogits},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


class Classification(Lightning):
    """
    Evaluator that is used for classification.

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
    train_dataloaders : DataLoader
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

    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 export_onnx: bool = True,
                 train_dataloader: Optional[DataLoader] = None,
                 **trainer_kwargs):
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        module = _ClassificationModule(criterion=criterion, learning_rate=learning_rate,
                                       weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)


@nni.trace
class _RegressionModule(_SupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: bool = True):
        super().__init__(criterion, {'mse': torchmetrics.MeanSquaredError},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


class Regression(Lightning):
    """
    Evaluator that is used for regression.

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
    train_dataloaders : DataLoader
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

    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 export_onnx: bool = True,
                 train_dataloader: Optional[DataLoader] = None,
                 **trainer_kwargs):
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        module = _RegressionModule(criterion=criterion, learning_rate=learning_rate,
                                   weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
