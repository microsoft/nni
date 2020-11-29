# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import random

import torch
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice

from ..interface import BaseOneShotTrainer
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


_logger = logging.getLogger(__name__)


class PathSamplingLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        self.op_names = []
        for name, module in layer_choice.named_modules():
            self.add_module(name, module)
        self.sampled = None  # sampled can be either a list of indices or an index

    def forward(self, *args, **kwargs):
        assert self.sampled, 'At least one path needs to be sampled before fprop.'
        if isinstance(self.sampled, list):
            return sum([getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled])
        else:
            return self.op_names[self.sampled](*args, **kwargs)

    def __length__(self):
        return len(self.op_names)


class PathSamplingInputChoice(nn.Module):
    def __init__(self, input_choice):
        self.num_candidates = input_choice.num_candidates
        self.sampled = None

    def forward(self, input_tensors):
        # TODO implement other ways including concatenate and single choice...
        return sum([input_tensors[i] for i in self.sampled])

    def __length__(self):
        return self.num_candidates


class SinglePathTrainer(BaseOneShotTrainer):
    """
    Single-path trainer. Samples a path every time and backpropagates on that path.

    Parameters
    ----------
    model : nn.Module
        Model with mutables.
    loss : callable
        Called with logits and targets. Returns a loss tensor.
    metrics : callable
        Returns a dict that maps metrics keys to metrics data.
    optimizer : Optimizer
        Optimizer that optimizes the model.
    num_epochs : int
        Number of epochs of training.
    train_loader : iterable
        Data loader of training. Raise ``StopIteration`` when one epoch is exhausted.
    dataset_valid : iterable
        Data loader of validation. Raise ``StopIteration`` when one epoch is exhausted.
    batch_size : int
        Batch size.
    workers: int
        Number of threads for data preprocessing. Not used for this trainer. Maybe removed in future.
    device : torch.device
        Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
        automatic detects GPU and selects GPU first.
    log_frequency : int
        Number of mini-batches to log metrics.
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.
    """

    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        self.sampling_modules = replace_layer_choice(self.model, PathSamplingLayerChoice)
        self.sampling_modules += replace_input_choice(self.mode, PathSamplingInputChoice)
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                        batch_size=batch_size,
                                                        num_workers=workers)

    def _resample(self):
        for module in self.sampling_modules:
            module.sampled = random.randint(0, len(module) - 1)
        # TODO: remember key and return

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            self.mutator.reset()
            logits = self.model(x)
            loss = self.loss(logits, y)
            loss.backward()
            self.optimizer.step()

            metrics = self.metrics(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.mutator.reset()
                logits = self.model(x)
                loss = self.loss(logits, y)
                metrics = self.metrics(logits, y)
                metrics["loss"] = loss.item()
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    _logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                 self.num_epochs, step + 1, len(self.valid_loader), meters)

    def export(self):
        return self._resample()


RandomTrainer = SinglePathTrainer
