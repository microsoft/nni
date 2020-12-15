# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random

import torch
import torch.nn as nn

from ..interface import BaseOneShotTrainer
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


_logger = logging.getLogger(__name__)


def _get_mask(sampled, total):
    multihot = [i == sampled or (isinstance(sampled, list) and i in sampled) for i in range(total)]
    return torch.tensor(multihot, dtype=torch.bool)  # pylint: disable=not-callable


class PathSamplingLayerChoice(nn.Module):
    """
    Mixed module, in which fprop is decided by exactly one or multiple (sampled) module.
    If multiple module is selected, the result will be sumed and returned.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, layer_choice):
        super(PathSamplingLayerChoice, self).__init__()
        self.op_names = []
        for name, module in layer_choice.named_children():
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self.sampled = None  # sampled can be either a list of indices or an index

    def forward(self, *args, **kwargs):
        assert self.sampled is not None, 'At least one path needs to be sampled before fprop.'
        if isinstance(self.sampled, list):
            return sum([getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return getattr(self, self.op_names[self.sampled])(*args, **kwargs)  # pylint: disable=invalid-sequence-index

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))


class PathSamplingInputChoice(nn.Module):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    sampled : int or list of int
        Sampled module indices.
    mask : tensor
        A multi-hot bool 1D-tensor representing the sampled mask.
    """

    def __init__(self, input_choice):
        super(PathSamplingInputChoice, self).__init__()
        self.n_candidates = input_choice.n_candidates
        self.n_chosen = input_choice.n_chosen
        self.sampled = None

    def forward(self, input_tensors):
        if isinstance(self.sampled, list):
            return sum([input_tensors[t] for t in self.sampled])  # pylint: disable=not-an-iterable
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))


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
    dataset_train : Dataset
        Dataset of training.
    dataset_valid : Dataset
        Dataset of validation.
    batch_size : int
        Batch size.
    workers: int
        Number of threads for data preprocessing. Not used for this trainer. Maybe removed in future.
    device : torch.device
        Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
        automatic detects GPU and selects GPU first.
    log_frequency : int
        Number of mini-batches to log metrics.
    """

    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                        batch_size=batch_size,
                                                        num_workers=workers)

    def _resample(self):
        result = {}
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = random.randint(0, len(module) - 1)
            module.sampled = result[name]
        return result

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            self._resample()
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

    def _validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                self._resample()
                logits = self.model(x)
                loss = self.loss(logits, y)
                metrics = self.metrics(logits, y)
                metrics["loss"] = loss.item()
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    _logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                 self.num_epochs, step + 1, len(self.valid_loader), meters)

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)
            self._validate_one_epoch(i)

    def export(self):
        return self._resample()


RandomTrainer = SinglePathTrainer
