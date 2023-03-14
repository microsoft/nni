# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Differentiable one-shot implementation."""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .base_lightning import BaseOneShotLightningModule
from .supermodule.differentiable import GumbelSoftmax
from .profiler import ExpectationProfilerPenalty

_logger = logging.getLogger(__name__)


class DartsLightningModule(BaseOneShotLightningModule):
    """Search implementation of :class:`~nni.nas.strategy.DARTS`.

    The dataloader is expected to be a CombinedLoader with two keys: ``train`` and ``val``.

    See Also
    --------
    nni.nas.strategy.DARTS
    nni.nas.pytorch.oneshot.base_lightning.BaseOneShotLightningModule
    """

    def __init__(self, training_module: pl.LightningModule, *,
                 arc_learning_rate: float = 3.0E-4,
                 gradient_clip_val: float | None = None,
                 log_prob_every_n_step: int = 10,
                 warmup_epochs: int = 0,
                 penalty: ExpectationProfilerPenalty | None = None):
        super().__init__(training_module)
        self.arc_learning_rate = arc_learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.log_prob_every_n_step = log_prob_every_n_step
        self.warmup_epochs = warmup_epochs

        assert penalty is None or isinstance(penalty, ExpectationProfilerPenalty)
        self.penalty = penalty

    def training_step(self, batch, batch_idx):
        # grad manually
        arc_optim = self.architecture_optimizers()
        if not isinstance(arc_optim, optim.Optimizer):
            raise TypeError(f'Expect arc_optim to be a single Optimizer, but found: {arc_optim}')

        # DARTS strategy makes sure that ``train`` and ``val`` must be in the batch
        trn_batch = batch['train']
        val_batch = batch['val']

        # Phase 1: architecture step. Only when warmup completes.
        if self.warmup_epochs == 0 or self.trainer.current_epoch >= self.warmup_epochs:
            # The _resample hook is kept for some darts-based NAS methods like proxyless.
            # See code of those methods for details.
            self.resample()
            arc_optim.zero_grad()
            arc_step_loss = self.training_module.training_step(val_batch, 2 * batch_idx)
            if isinstance(arc_step_loss, dict):
                arc_step_loss = arc_step_loss['loss']

            if self.penalty is not None:
                arc_step_loss, details = self.penalty(arc_step_loss, self.export_probs())
                self.log_dict({f'penalty/{k}': v for k, v in details.items()})

            self.manual_backward(arc_step_loss)
            arc_optim.step()

        # Phase 2: model step
        self.resample()
        loss_and_metrics = self.training_module.training_step(trn_batch, 2 * batch_idx + 1)
        w_step_loss = loss_and_metrics['loss'] if isinstance(loss_and_metrics, dict) else loss_and_metrics
        self.advance_optimization(w_step_loss, batch_idx, self.gradient_clip_val)

        # Update learning rates
        self.advance_lr_schedulers(batch_idx)

        if batch_idx % self.log_prob_every_n_step == 0:
            # NOTE: Ideally we should use global_step, but, who cares.
            self.log_probs(self.export_probs())

        return loss_and_metrics

    def arch_parameters(self) -> list[nn.Parameter]:
        # The alpha in DartsXXXChoices are the architecture parameters of DARTS. They share one optimizer.
        ctrl_params = []
        for m in self.supernet_modules():
            if hasattr(m, 'arch_parameters'):
                ctrl_params += list(m.arch_parameters())  # type: ignore
        # Follow the hyper-parameters used in
        # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/architect.py#L17
        return list(set(ctrl_params))

    def on_train_epoch_start(self):
        if self.trainer.current_epoch < self.warmup_epochs:
            # Still warming up
            for param in self.arch_parameters():
                param.requires_grad_(False)
        elif self.trainer.current_epoch >= self.warmup_epochs > 0:
            # Warmed up, and requires_grad is once False
            for param in self.arch_parameters():
                param.requires_grad_()
        return super().on_train_epoch_start()

    def postprocess_weight_optimizers(self, optimizer):
        blacklist = set(self.arch_parameters())

        def _process_fn(opt):
            if isinstance(opt, optim.Optimizer):
                for param_group in opt.param_groups:
                    if 'params' in param_group:
                        # Remove some parameters.
                        param_group['params'] = [p for p in param_group['params'] if p not in blacklist]
                return opt
            elif isinstance(opt, list):
                return [_process_fn(o) for o in opt]
            elif isinstance(opt, tuple):
                return tuple([_process_fn(o) for o in opt])
            elif isinstance(opt, dict):
                return {k: _process_fn(v) for k, v in opt.items()}
            else:
                return opt

        return _process_fn(optimizer)

    def configure_architecture_optimizers(self):
        ctrl_params = self.arch_parameters()
        if not ctrl_params:
            raise ValueError('No architecture parameters found. Nothing to search.')
        ctrl_optim = torch.optim.Adam(ctrl_params, self.arc_learning_rate, betas=(0.5, 0.999), weight_decay=1.0E-3)

        return ctrl_optim


class GumbelDartsLightningModule(DartsLightningModule):
    """Extend :class:`DartsLightningModule` to support gumbel-softmax with temperature annealing.

    The default implementation of :class:`~nni.nas.strategy.GumbelDARTS`.

    See Also
    --------
    nni.nas.strategy.GumbelDARTS
    DartsLightningModule
    """

    def __init__(self, training_module: pl.LightningModule, temperature_scheduler: LinearTemperatureScheduler, *args, **kwargs):
        super().__init__(training_module, *args, **kwargs)
        self.temperature_scheduler = temperature_scheduler

    def on_train_epoch_start(self):
        temp = self.temperature_scheduler.step(self.trainer.current_epoch, self.trainer.max_epochs)
        self.log('gumbel_temperature', temp)

        for module in self.modules():
            if isinstance(module, GumbelSoftmax):
                module.tau = temp  # type: ignore

        return super().on_train_epoch_start()


class LinearTemperatureScheduler:
    """
    Linear temperature scheduler to support temperature annealing in :class:`GumbelDartsLightningModule`.
    Temperature decreases from ``init`` to ``min`` linearly throughout ``total`` given in :meth:`step`.

    Parameters
    ----------
    init
        Initial temperature.
    min
        Minimum temperature.
    """

    def __init__(self, init: float, min: float):  # pylint: disable=redefined-builtin
        if not isinstance(init, float) and isinstance(min, float):  # pylint: disable=redefined-builtin
            raise TypeError('init and min must be float')
        if not (init >= min >= 0):
            raise ValueError('Invalid temperature range: init >= min >= 0')

        self.init = init
        self.min = min

    def step(self, current: int, total: int | None = None):
        """Compute temperature for current epoch.

        ``current`` is 0-indexed in the range of [0, total).
        If ``total`` is not given, ``init`` must be equal to ``min``.
        """
        if total is None:
            if self.init == self.min:
                return self.min
            else:
                raise ValueError('Total epoch is None, but temperature is not fixed.')
        if current > total:
            _logger.warning('Current epoch (%d) is larger than total epoch (%d). Assuming current = total.', current, total)
            current = total
        return (1 - current / total) * (self.init - self.min) + self.min
