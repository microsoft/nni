# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['MultiModelLightningModule', 'MultiModelTrainer']

import pytorch_lightning
import torch
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch import nn
from torchmetrics import Metric

import nni
from nni.nas.evaluator.pytorch.lightning import LightningModule


class MultiModelLightningModule(LightningModule):
    """The lightning module for a merged "multi-model".

    The output of the multi-model is expected to be a tuple of tensors.
    The tensors will be each passed to a criterion and a metric.
    The loss will be added up for back propagation, and the metrics will be logged.

    The reported metric will be a list of metrics, one for each model.

    Parameters
    ----------
    criterion
        Loss function.
    metric
        Metric function.
    n_models
        Number of models in the multi-model.
    """

    def __init__(self, criterion: nn.Module, metric: Metric, n_models: int | None = None):
        super().__init__()
        self.criterion = criterion
        self.metric = metric
        self.n_models = n_models

    def _dump(self) -> dict:
        return {
            'criterion': self.criterion,
            'metric': self.metric,
            'n_models': self.n_models,
        }

    @staticmethod
    def _load(criterion: nn.Module, metric: Metric, n_models: int | None = None) -> MultiModelLightningModule:
        return MultiModelLightningModule(criterion, metric, n_models)

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
            self.log(f'train_metric_{idx}', self.metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)
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
            self.log(f'val_metric_{idx}', self.metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

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
            self.log(f'test_metric_{idx}', self.metric(y_hat.to("cpu"), y.to("cpu")), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_validation_epoch_end(self):
        nni.report_intermediate_result(self._get_validation_metrics())  # type: ignore

    def teardown(self, stage):
        if stage == 'fit':
            nni.report_final_result(self._get_validation_metrics())  # type: ignore

    def _get_validation_metrics(self):
        # TODO: split metric of multiple models?
        assert self.n_models is not None
        return [self.trainer.callback_metrics[f'val_metric_{idx}'].item() for idx in range(self.n_models)]


class _BypassStrategy(SingleDeviceStrategy):
    strategy_name = "single_device"

    def model_to_device(self) -> None:
        pass


@nni.trace
class MultiModelTrainer(pytorch_lightning.Trainer):
    """
    Trainer for cross-graph optimization.

    Parameters
    ----------
    use_cgo
        Whether cross-graph optimization (CGO) is used.
        If it is True, CGO will manage device placement.
        Any device placement from pytorch lightning will be bypassed.
        default: False
    trainer_kwargs
        Optional keyword arguments passed to trainer. See
        `Lightning documentation <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for details.
    """

    def __init__(self, use_cgo: bool = True, **trainer_kwargs):
        if use_cgo:
            # Accelerator and strategy can be both set at lightning 2.0.
            # if "accelerator" in trainer_kwargs:
            #     raise ValueError("accelerator should not be set when cross-graph optimization is enabled.")

            if 'strategy' in trainer_kwargs:
                raise ValueError("MultiModelTrainer does not support specifying strategy")

            trainer_kwargs['strategy'] = _BypassStrategy()

        super().__init__(**trainer_kwargs)
