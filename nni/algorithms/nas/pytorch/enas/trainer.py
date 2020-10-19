# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim

from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup, to_device
from .mutator import EnasMutator

logger = logging.getLogger(__name__)


class EnasTrainer(Trainer):
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
    dataset_train : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    dataset_valid : Dataset
        Dataset for testing.
    mutator : EnasMutator
        Use when customizing your own mutator or a mutator with customized parameters.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    callbacks : list of Callback
        list of callbacks to trigger at events.
    entropy_weight : float
        Weight of sample entropy loss.
    skip_weight : float
        Weight of skip penalty loss.
    baseline_decay : float
        Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
    child_steps : int
        How many mini-batches for model training per epoch.
    mutator_lr : float
        Learning rate for RL controller.
    mutator_steps_aggregate : int
        Number of steps that will be aggregated into one mini-batch for RL controller.
    mutator_steps : int
        Number of mini-batches for each epoch of RL controller learning.
    aux_weight : float
        Weight of auxiliary head loss. ``aux_weight * aux_loss`` will be added to total loss.
    test_arc_per_epoch : int
        How many architectures are chosen for direct test after each epoch.
    """
    def __init__(self, model, loss, metrics, reward_function,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None, callbacks=None,
                 entropy_weight=0.0001, skip_weight=0.8, baseline_decay=0.999, child_steps=500,
                 mutator_lr=0.00035, mutator_steps_aggregate=20, mutator_steps=50, aux_weight=0.4,
                 test_arc_per_epoch=1):
        super().__init__(model, mutator if mutator is not None else EnasMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)
        self.reward_function = reward_function
        self.mutator_optim = optim.Adam(self.mutator.parameters(), lr=mutator_lr)
        self.batch_size = batch_size
        self.workers = workers

        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.
        self.mutator_steps_aggregate = mutator_steps_aggregate
        self.mutator_steps = mutator_steps
        self.child_steps = child_steps
        self.aux_weight = aux_weight
        self.test_arc_per_epoch = test_arc_per_epoch

        self.init_dataloader()

    def init_dataloader(self):
        n_train = len(self.dataset_train)
        split = n_train // 10
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.workers)
        self.train_loader = cycle(self.train_loader)
        self.valid_loader = cycle(self.valid_loader)

    def train_one_epoch(self, epoch):
        # Sample model and train
        self.model.train()
        self.mutator.eval()
        meters = AverageMeterGroup()
        for step in range(1, self.child_steps + 1):
            x, y = next(self.train_loader)
            x, y = to_device(x, self.device), to_device(y, self.device)
            self.optimizer.zero_grad()

            with torch.no_grad():
                self.mutator.reset()
            self._write_graph_status()
            logits = self.model(x)

            if isinstance(logits, tuple):
                logits, aux_logits = logits
                aux_loss = self.loss(aux_logits, y)
            else:
                aux_loss = 0.
            metrics = self.metrics(logits, y)
            loss = self.loss(logits, y)
            loss = loss + self.aux_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            metrics["loss"] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Model Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1,
                            self.num_epochs, step, self.child_steps, meters)

        # Train sampler (mutator)
        self.model.eval()
        self.mutator.train()
        meters = AverageMeterGroup()
        for mutator_step in range(1, self.mutator_steps + 1):
            self.mutator_optim.zero_grad()
            for step in range(1, self.mutator_steps_aggregate + 1):
                x, y = next(self.valid_loader)
                x, y = to_device(x, self.device), to_device(y, self.device)

                self.mutator.reset()
                with torch.no_grad():
                    logits = self.model(x)
                self._write_graph_status()
                metrics = self.metrics(logits, y)
                reward = self.reward_function(logits, y)
                if self.entropy_weight:
                    reward += self.entropy_weight * self.mutator.sample_entropy.item()
                self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                loss = self.mutator.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.mutator.sample_skip_penalty
                metrics["reward"] = reward
                metrics["loss"] = loss.item()
                metrics["ent"] = self.mutator.sample_entropy.item()
                metrics["log_prob"] = self.mutator.sample_log_prob.item()
                metrics["baseline"] = self.baseline
                metrics["skip"] = self.mutator.sample_skip_penalty

                loss /= self.mutator_steps_aggregate
                loss.backward()
                meters.update(metrics)

                cur_step = step + (mutator_step - 1) * self.mutator_steps_aggregate
                if self.log_frequency is not None and cur_step % self.log_frequency == 0:
                    logger.info("RL Epoch [%d/%d] Step [%d/%d] [%d/%d]  %s", epoch + 1, self.num_epochs,
                                mutator_step, self.mutator_steps, step, self.mutator_steps_aggregate,
                                meters)

            nn.utils.clip_grad_norm_(self.mutator.parameters(), 5.)
            self.mutator_optim.step()

    def validate_one_epoch(self, epoch):
        with torch.no_grad():
            for arc_id in range(self.test_arc_per_epoch):
                meters = AverageMeterGroup()
                for x, y in self.test_loader:
                    x, y = to_device(x, self.device), to_device(y, self.device)
                    self.mutator.reset()
                    logits = self.model(x)
                    if isinstance(logits, tuple):
                        logits, _ = logits
                    metrics = self.metrics(logits, y)
                    loss = self.loss(logits, y)
                    metrics["loss"] = loss.item()
                    meters.update(metrics)

                logger.info("Test Epoch [%d/%d] Arc [%d/%d] Summary  %s",
                            epoch + 1, self.num_epochs, arc_id + 1, self.test_arc_per_epoch,
                            meters.summary())
