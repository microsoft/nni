# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Union, Optional, List, Type

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torchmetrics
import torchmetrics.classification
import torch.utils.data as torch_data

import nni
from nni.common.serializer import is_traceable
from nni.nas.evaluator import MutableEvaluator

__all__ = [
    'LightningModule', 'Trainer', 'DataLoader', 'Lightning', 'Classification', 'Regression',
    'SupervisedLearningModule', 'ClassificationModule', 'RegressionModule',
]

_logger = logging.getLogger(__name__)


class LightningModule(pl.LightningModule):
    """
    Basic wrapper of generated model.
    Lightning modules used in NNI should inherit this class.

    It's a subclass of ``pytorch_lightning.LightningModule``.
    See https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

    See :class:`SupervisedLearningModule` as an example.
    """

    @property
    def model(self) -> nn.Module:
        """The inner model (architecture) to train / evaluate.

        It will be only available after calling :meth:`set_model`.
        """
        model = getattr(self, '_model', None)
        if model is None:
            raise RuntimeError('Model is not set. Please call set_model() first.')
        return model

    def set_model(self, model: nn.Module) -> None:
        """Set the inner model (architecture) to train / evaluate.

        As there is no explicit method to "unset" a model,
        the model is left in the lightning module after the method is called.
        We don't recommend relying on this behavior.
        """
        if not isinstance(model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')
        self._model = model


Trainer = nni.trace(pl.Trainer)
Trainer.__doc__ = """
Traced version of ``pytorch_lightning.Trainer``. See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
"""
DataLoader = nni.trace(torch_data.DataLoader)
DataLoader.__doc__ = """
Traced version of ``torch.utils.data.DataLoader``. See https://pytorch.org/docs/stable/data.html
"""


@nni.trace
class Lightning(MutableEvaluator):
    """
    Delegate the whole training to PyTorch Lightning.

    Since the arguments passed to the initialization needs to be serialized, ``LightningModule``, ``Trainer`` or
    ``DataLoader`` in this file should be used. Another option is to hide dataloader in the Lightning module, in
    which case, dataloaders are not required for this class to work.

    Following the programming style of Lightning, metrics sent to NNI should be obtained from ``callback_metrics``
    in trainer. Two hooks are added at the end of validation epoch and the end of ``fit``, respectively. The metric name
    and type depend on the specific task.

    .. warning::

       The Lightning evaluator are stateful. If you try to use a previous Lightning evaluator,
       please note that the inner ``lightning_module`` and ``trainer`` will be reused.

    Parameters
    ----------
    lightning_module
        Lightning module that defines the training logic.
    trainer
        Lightning trainer that handles the training.
    train_dataloders
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
        It can be any types of dataloader supported by Lightning.
    val_dataloaders
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
        It can be any types of dataloader supported by Lightning.
    datamodule
        Used in ``trainer.fit()``. See `Lightning DataModule <https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html>`__.
    fit_kwargs
        Keyword arguments passed to ``trainer.fit()``.
    detect_interrupt
        Lightning has a `graceful shutdown <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__
        mechanism. It does not terminate the whole program (but only the training) when a KeyboardInterrupt is received.
        Setting this to ``True`` will raise the KeyboardInterrupt to the main process, so that the whole program can be terminated.

    Examples
    --------
    Users should define a Lightning module that inherits :class:`LightningModule`,
    and use :class:`Trainer` and :class:`DataLoader` from ```nni.nas.evaluator.pytorch``,
    and make them parameters of this evaluator::

        import nni
        from nni.nas.evaluator.pytorch.lightning import Lightning, LightningModule, Trainer, DataLoader
    """

    def __init__(self, lightning_module: LightningModule, trainer: Trainer,
                 train_dataloaders: Optional[Any] = None,
                 val_dataloaders: Optional[Any] = None,
                 train_dataloader: Optional[Any] = None,
                 datamodule: Optional[pl.LightningDataModule] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 detect_interrupt: bool = True):
        assert isinstance(lightning_module, LightningModule), f'Lightning module must be an instance of {__name__}.LightningModule.'
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        if not (isinstance(trainer, pl.Trainer) and is_traceable(trainer)):
            raise TypeError(f'Trainer must be imported from {__name__}, but found {trainer.__class__.__qualname__}')
        if not _check_dataloader(train_dataloaders):
            warnings.warn(f'When using training service to spawn trials, please try to wrap PyTorch DataLoader with nni.trace or '
                          f'import DataLoader from {__name__}: {train_dataloaders}',
                          RuntimeWarning)
        if not _check_dataloader(val_dataloaders):
            warnings.warn(f'When using training service to spawn trials, please try to wrap PyTorch DataLoader with nni.trace or '
                          f'import DataLoader from {__name__}: {val_dataloaders}',
                          RuntimeWarning)
        self.module = lightning_module
        self.trainer = trainer
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.datamodule = datamodule
        self.fit_kwargs = fit_kwargs or {}
        self.detect_interrupt = detect_interrupt

    def evaluate(self, model):
        """
        Fit the model with provided dataloader, with Lightning trainer.
        If ``train_dataloaders`` is not provided, ``trainer.validate()`` will be called.

        Parameters
        ----------
        model
            The model to fit.
        """
        if self.is_mutable():
            raise RuntimeError('Mutable evaluator must first be `freeze()` before evaluation.')

        self.module.set_model(model)
        if self.datamodule is not None:
            _logger.info('Fit with datamodule. Train and valid dataloaders will be ignored.')
            rv = self.trainer.fit(self.module, self.datamodule, **self.fit_kwargs)
        elif self.train_dataloaders is None and self.val_dataloaders is not None:
            _logger.info('Only validation dataloaders are available. Skip to validation.')
            rv = self.trainer.validate(self.module, self.val_dataloaders, **self.fit_kwargs)
        else:
            if self.val_dataloaders is None:
                _logger.warning('Validation dataloaders are missing. Safe to ignore this warning when using one-shot strategy.')
            rv = self.trainer.fit(self.module, self.train_dataloaders, self.val_dataloaders, **self.fit_kwargs)

        if self.detect_interrupt:
            from pytorch_lightning.trainer.states import TrainerStatus
            if self.trainer.state.status == TrainerStatus.INTERRUPTED:
                _logger.warning('Trainer status is detected to be interrupted.')
                raise KeyboardInterrupt('Trainer status is detected to be interrupted.')

        return rv

    @property
    def train_dataloader(self):
        warnings.warn('train_dataloader is deprecated, please use `train_dataloaders`.', DeprecationWarning)

    def __eq__(self, other):
        if not isinstance(other, Lightning):
            return False
        return self.module == other.module and self.trainer == other.trainer and \
            self.train_dataloaders == other.train_dataloaders and self.val_dataloaders == other.val_dataloaders and \
            self.fit_kwargs == other.fit_kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module}, {self.trainer}, train_dataloaders={self.train_dataloaders}, ' \
            f'val_dataloaders={self.val_dataloaders}, fit_kwargs={self.fit_kwargs})'

    def fit(self, model):
        warnings.warn('`fit` is deprecated, please use `evaluate`.', DeprecationWarning)
        return self.evaluate(model)


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

class SupervisedLearningModule(LightningModule):

    trainer: pl.Trainer

    def __init__(self, criterion: Type[nn.Module], metrics: Dict[str, torchmetrics.Metric],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: Union[Path, str, bool, None] = None):
        super().__init__()
        self.save_hyperparameters('criterion', 'optimizer', 'learning_rate', 'weight_decay')
        self.criterion = criterion()
        self.optimizer = optimizer
        self.metrics = nn.ModuleDict(metrics)

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

        if self.export_onnx is not None:
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
        if nni.get_current_parameter() is not None and not self.trainer.sanity_checking:
            # Don't report metric when sanity checking
            nni.report_intermediate_result(self._get_result_for_report())

    def on_fit_end(self):
        # Inline import to avoid errors with unsupported lightning version
        from pytorch_lightning.trainer.states import TrainerFn
        if self.trainer.state.fn == TrainerFn.FITTING:
            self._final_report()

    def on_validation_end(self):
        from pytorch_lightning.trainer.states import TrainerFn
        if self.trainer.state.fn == TrainerFn.VALIDATING:
            self._final_report()

    def _final_report(self):
        if nni.get_current_parameter() is not None:
            nni.report_final_result(self._get_result_for_report())

    def _get_result_for_report(self):
        stage = 'val'
        if not self.trainer.val_dataloaders:
            _logger.debug('No validation dataloader. Use results on training set instead.')
            stage = 'train'

        if len(self.metrics) == 1:
            metric_name = next(iter(self.metrics))
            return self.trainer.callback_metrics[f'{stage}_{metric_name}'].item()
        else:
            warnings.warn('Multiple metrics without "default" is not supported by current framework.')
            return {name: self.trainer.callback_metrics[f'{stage}_{name}'].item() for name in self.metrics}


class _AccuracyWithLogits(torchmetrics.Accuracy):
    # Only for torchmetrics < 0.11
    def update(self, pred, target):
        return super().update(nn_functional.softmax(pred, dim=-1), target)  # type: ignore


@nni.trace
class ClassificationModule(SupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: bool = False,
                 num_classes: Optional[int] = None):

        from packaging.version import Version
        if Version(torchmetrics.__version__) < Version('0.11.0'):
            # Older version accepts num_classes = None
            metrics = {'acc': _AccuracyWithLogits()}  # type: ignore # pylint: disable=no-value-for-parameter
        else:
            if num_classes is None:
                raise ValueError('num_classes must be specified for torchmetrics >= 0.11. '
                                 'Please either specify it or use an older version of torchmetrics.')
            metrics = {'acc': torchmetrics.Accuracy('multiclass', num_classes=num_classes)}

        super().__init__(criterion, metrics,  # type: ignore
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


@nni.trace
class Classification(Lightning):
    """
    Evaluator that is used for classification.

    Available callback metrics in :class:`Classification` are:

    - train_loss
    - train_acc
    - val_loss
    - val_acc

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
    datamodule
        Used in ``trainer.fit()``. See `Lightning DataModule <https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html>`__.
    export_onnx : bool
        If true, model will be exported to ``model.onnx`` before training starts. default true
    num_classes : int
        Number of classes for classification task.
        Required for torchmetrics >= 0.11.0. default: None
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.

    Examples
    --------
    >>> evaluator = Classification()

    To use customized criterion and optimizer:

    >>> evaluator = Classification(nn.LabelSmoothingCrossEntropy, optimizer=torch.optim.SGD)

    Extra keyword arguments will be passed to trainer, some of which might be necessary to enable GPU acceleration:

    >>> evaluator = Classification(accelerator='gpu', devices=2, strategy='ddp')
    """

    def __init__(self, criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 datamodule: Optional[pl.LightningDataModule] = None,
                 export_onnx: bool = False,
                 train_dataloader: Optional[DataLoader] = None,
                 num_classes: Optional[int] = None,
                 **trainer_kwargs):
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        module = ClassificationModule(criterion=criterion, learning_rate=learning_rate,
                                      weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx,
                                      num_classes=num_classes)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                         datamodule=datamodule)


@nni.trace
class RegressionModule(SupervisedLearningModule):
    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 export_onnx: bool = False):
        super().__init__(criterion, {'mse': torchmetrics.MeanSquaredError()},
                         learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer,
                         export_onnx=export_onnx)


@nni.trace
class Regression(Lightning):
    """
    Evaluator that is used for regression.

    Available callback metrics in :class:`Regression` are:

    - train_loss
    - train_mse
    - val_loss
    - val_mse

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
    datamodule
        Used in ``trainer.fit()``. See `Lightning DataModule <https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html>`__.
    export_onnx : bool
        If true, model will be exported to ``model.onnx`` before training starts. default: true
    trainer_kwargs : dict
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.

    Examples
    --------
    >>> evaluator = Regression()

    Extra keyword arguments will be passed to trainer, some of which might be necessary to enable GPU acceleration:

    >>> evaluator = Regression(gpus=1)
    """

    def __init__(self, criterion: Type[nn.Module] = nn.MSELoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
                 datamodule: Optional[pl.LightningDataModule] = None,
                 export_onnx: bool = False,
                 train_dataloader: Optional[DataLoader] = None,
                 **trainer_kwargs):
        if train_dataloader is not None:
            warnings.warn('`train_dataloader` is deprecated and replaced with `train_dataloaders`.', DeprecationWarning)
            train_dataloaders = train_dataloader
        module = RegressionModule(criterion=criterion, learning_rate=learning_rate,
                                  weight_decay=weight_decay, optimizer=optimizer, export_onnx=export_onnx)
        super().__init__(module, Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders,
                         datamodule=datamodule)


# Alias for backwards compatibility
_SupervisedLearningModule = SupervisedLearningModule
_ClassificationModule = ClassificationModule
_RegressionModule = RegressionModule
