# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, Dict, Union, Optional, List


import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment


import nni

from .lightning import LightningModule
from ...serializer import serialize_cls


@serialize_cls
class _MultiModelSupervisedLearningModule(LightningModule):
    def __init__(self, criterion: nn.Module, metrics: Dict[str, pl.metrics.Metric],
                 n_models: int = 0,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam):
        super().__init__()
        self.save_hyperparameters('criterion', 'optimizer', 'learning_rate', 'weight_decay')
        self.criterion = criterion()
        self.criterion_cls = criterion
        self.optimizer = optimizer
        self.metrics = nn.ModuleDict({name: cls() for name, cls in metrics.items()})
        self.n_models = n_models

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        multi_y_hat = self(x)
        assert(len(multi_y_hat) == self.n_models)
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
        assert(len(multi_y_hat) == self.n_models)
        for idx, y_hat in enumerate(multi_y_hat):
            self.log(f'val_loss_{idx}', self.criterion(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
            for name, metric in self.metrics.items():
                self.log(f'val_{idx}_' + name, metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        multi_y_hat = self(x)
        assert(len(multi_y_hat) == self.n_models)
        for idx, y_hat in enumerate(multi_y_hat):
            self.log(f'test_loss_{idx}', self.criterion(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
            for name, metric in self.metrics.items():
                self.log(f'test_{idx}_' + name, metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self._get_validation_metrics())

    def teardown(self, stage):
        if stage == 'fit':
            nni.report_final_result(self._get_validation_metrics())

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

    def __init__(self, criterion: nn.Module, metrics: Dict[str, pl.metrics.Metric],
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam):
        super().__init__(criterion, metrics, learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)


class BypassPlugin(TrainingTypePlugin):
    """ Plugin that handles communication on a single device. """

    def __init__(self, device: str):
        super().__init__()
        self.device: str = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @property
    def on_tpu(self) -> bool:
        return False

    @property
    def on_gpu(self) -> bool:
        return "cuda" in self.device and torch.cuda.is_available()

    def reduce(self, tensor: Union[Any, torch.Tensor], *args: Any, **kwargs: Any) -> Union[Any, torch.Tensor]:
        """
        Reduces a tensor from several distributed processes to one aggregated tensor.
        As this plugin only operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes """
        return tensor

    @property
    def root_device(self) -> torch.device:
        return torch.device(self.device)

    def model_to_device(self) -> None:
        if self.on_gpu:
            torch.cuda.set_device(self.root_device)

        self._model.to(self.root_device)

    def setup(self, model: torch.nn.Module) -> torch.nn.Module:
        self.model_to_device()
        return self.model

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def setup(self, model: torch.nn.Module) -> torch.nn.Module:
        # self.model_to_device()
        return self.model


def get_accelerator_connector(
        num_processes: int = 1,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        distributed_backend: Optional[str] = None,
        auto_select_gpus: bool = False,
        gpus: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        sync_batchnorm: bool = False,
        benchmark: bool = False,
        replace_sampler_ddp: bool = True,
        deterministic: bool = False,
        precision: int = 32,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None):
    return AcceleratorConnector(
        num_processes, tpu_cores, distributed_backend, auto_select_gpus, gpus, num_nodes, sync_batchnorm, benchmark,
        replace_sampler_ddp, deterministic, precision, amp_backend, amp_level, plugins
    )


@serialize_cls
class BypassAccelerator(Accelerator):
    def __init__(self, precision_plugin=None, device="cuda:0"):
        if precision_plugin is None:
            precision_plugin = get_accelerator_connector().precision_plugin
        super().__init__(precision_plugin=precision_plugin, training_type_plugin=BypassPlugin(device))


# @serialize_cls
# class _ClassificationModule(_MultiModelSupervisedLearningModule):
#     def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss,
#                  learning_rate: float = 0.001,
#                  weight_decay: float = 0.,
#                  optimizer: optim.Optimizer = optim.Adam):
#         super().__init__(criterion, {'acc': _AccuracyWithLogits},
#                          learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)

# class Classification(Lightning):
#     """
#     Trainer that is used for classification.

#     Parameters
#     ----------
#     criterion : nn.Module
#         Class for criterion module (not an instance). default: ``nn.CrossEntropyLoss``
#     learning_rate : float
#         Learning rate. default: 0.001
#     weight_decay : float
#         L2 weight decay. default: 0
#     optimizer : Optimizer
#         Class for optimizer (not an instance). default: ``Adam``
#     train_dataloders : DataLoader
#         Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
#         If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
#     val_dataloaders : DataLoader or List of DataLoader
#         Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
#         If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
#     trainer_kwargs : dict
#         Optional keyword arguments passed to trainer. See
#         `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`__ for details.
#     """

#     def __init__(self, criterion: nn.Module = nn.CrossEntropyLoss,
#                  learning_rate: float = 0.001,
#                  weight_decay: float = 0.,
#                  optimizer: optim.Optimizer = optim.Adam,
#                  train_dataloader: Optional[DataLoader] = None,
#                  val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
#                  **trainer_kwargs):
#         module = _ClassificationModule(criterion=criterion, learning_rate=learning_rate,
#                                        weight_decay=weight_decay, optimizer=optimizer)
#         super().__init__(module, Trainer(**trainer_kwargs),
#                          train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)


# @serialize_cls
# class _RegressionModule(_SupervisedLearningModule):
#     def __init__(self, criterion: nn.Module = nn.MSELoss,
#                  learning_rate: float = 0.001,
#                  weight_decay: float = 0.,
#                  optimizer: optim.Optimizer = optim.Adam):
#         super().__init__(criterion, {'mse': pl.metrics.MeanSquaredError},
#                          learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)


# class Regression(Lightning):
#     """
#     Trainer that is used for regression.

#     Parameters
#     ----------
#     criterion : nn.Module
#         Class for criterion module (not an instance). default: ``nn.MSELoss``
#     learning_rate : float
#         Learning rate. default: 0.001
#     weight_decay : float
#         L2 weight decay. default: 0
#     optimizer : Optimizer
#         Class for optimizer (not an instance). default: ``Adam``
#     train_dataloders : DataLoader
#         Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
#         If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
#     val_dataloaders : DataLoader or List of DataLoader
#         Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
#         If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
#     trainer_kwargs : dict
#         Optional keyword arguments passed to trainer. See
#         `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`__ for details.
#     """

#     def __init__(self, criterion: nn.Module = nn.MSELoss,
#                  learning_rate: float = 0.001,
#                  weight_decay: float = 0.,
#                  optimizer: optim.Optimizer = optim.Adam,
#                  train_dataloader: Optional[DataLoader] = None,
#                  val_dataloaders: Union[DataLoader, List[DataLoader], None] = None,
#                  **trainer_kwargs):
#         module = _RegressionModule(criterion=criterion, learning_rate=learning_rate,
#                                    weight_decay=weight_decay, optimizer=optimizer)
#         super().__init__(module, Trainer(**trainer_kwargs),
#                          train_dataloader=train_dataloader, val_dataloaders=val_dataloaders)
