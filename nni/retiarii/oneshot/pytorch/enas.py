# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader

from nni.nas.oneshot.pytorch.enas import ReinforceController, ReinforceField

from ..interface import BaseOneShotTrainer
from .random import PathSamplingLayerChoice, PathSamplingInputChoice
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device

_logger = logging.getLogger(__name__)


class EnasTrainer(BaseOneShotTrainer):
    """
    ENAS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    reward_function : callable
        Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    ctrl_lr : float
        Learning rate for RL controller.
    ctrl_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    ctrl_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    ctrl_kwargs : dict
        Optional kwargs that will be passed to :class:`ReinforceController`.
    """

    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset,
                 batch_size=64, workers=4, device=None, log_frequency=None,
                 grad_clip=5., entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999,
                 ctrl_lr=0.00035, ctrl_steps_aggregate=20, ctrl_kwargs=None):
        warnings.warn('EnasTrainer is deprecated. Please use strategy.ENAS instead.', DeprecationWarning)

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        self.nas_modules = []
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)
        self.model.to(self.device)

        self.nas_fields = [ReinforceField(name, len(module),
                                          isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1)
                           for name, module in self.nas_modules]
        self.controller = ReinforceController(self.nas_fields, **(ctrl_kwargs or {}))

        self.grad_clip = grad_clip
        self.reward_function = reward_function
        self.ctrl_optim = optim.Adam(self.controller.parameters(), lr=ctrl_lr)
        self.batch_size = batch_size
        self.workers = workers

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.ctrl_steps_aggregate = ctrl_steps_aggregate

        self.init_dataloader()

    def init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = SubsetRandomSampler(indices[:-split])
        valid_sampler = SubsetRandomSampler(indices[-split:])
        self.train_loader = DataLoader(self.dataset,
                                       batch_size=self.batch_size,
                                       sampler=train_sampler,
                                       num_workers=self.workers)
        self.valid_loader = DataLoader(self.dataset,
                                       batch_size=self.batch_size,
                                       sampler=valid_sampler,
                                       num_workers=self.workers)

    def _train_model(self, epoch):
        self.model.train()
        self.controller.eval()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
            x, y = to_device(x, self.device), to_device(y, self.device)
            self.optimizer.zero_grad()

            self._resample()
            logits = self.model(x)
            metrics = self.metrics(logits, y)
            loss = self.loss(logits, y)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            metrics['loss'] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info('Model Epoch [%d/%d] Step [%d/%d]  %s', epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        meters = AverageMeterGroup()
        self.ctrl_optim.zero_grad()
        for ctrl_step, (x, y) in enumerate(self.valid_loader):
            x, y = to_device(x, self.device), to_device(y, self.device)

            self._resample()
            with torch.no_grad():
                logits = self.model(x)
            metrics = self.metrics(logits, y)
            reward = self.reward_function(logits, y)
            if self.entropy_weight:
                reward += self.entropy_weight * self.controller.sample_entropy.item()  # type: ignore
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
            loss = self.controller.sample_log_prob * (reward - self.baseline)
            if self.skip_weight:
                loss += self.skip_weight * self.controller.sample_skip_penalty
            metrics['reward'] = reward
            metrics['loss'] = loss.item()
            metrics['ent'] = self.controller.sample_entropy.item()  # type: ignore
            metrics['log_prob'] = self.controller.sample_log_prob.item()  # type: ignore
            metrics['baseline'] = self.baseline
            metrics['skip'] = self.controller.sample_skip_penalty

            loss /= self.ctrl_steps_aggregate
            loss.backward()
            meters.update(metrics)

            if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
                self.ctrl_optim.step()
                self.ctrl_optim.zero_grad()

            if self.log_frequency is not None and ctrl_step % self.log_frequency == 0:
                _logger.info('RL Epoch [%d/%d] Step [%d/%d]  %s', epoch + 1, self.num_epochs,
                             ctrl_step + 1, len(self.valid_loader), meters)

    def _resample(self):
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def fit(self):
        for i in range(self.num_epochs):
            self._train_model(i)
            self._train_controller(i)

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()
