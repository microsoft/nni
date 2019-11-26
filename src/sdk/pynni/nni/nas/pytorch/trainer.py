# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod

import torch

from .base_trainer import BaseTrainer

_logger = logging.getLogger(__name__)


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, torch.Tensor):
            olist = o.tolist()
            if "bool" not in o.type().lower() and all(map(lambda d: d == 0 or d == 1, olist)):
                _logger.warning("Every element in %s is either 0 or 1. "
                                "You might consider convert it into bool.", olist)
            return olist
        return super().default(o)


class Trainer(BaseTrainer):
    def __init__(self, model, mutator, loss, metrics, optimizer, num_epochs,
                 dataset_train, dataset_valid, batch_size, workers, device, log_frequency, callbacks):
        """
        Trainer initialization.

        Parameters
        ----------
        model : nn.Module
            Model with mutables.
        mutator : BaseMutator
            A mutator object that has been initialized with the model.
        loss : callable
            Called with logits and targets. Returns a loss tensor.
        metrics : callable
            Returns a dict that maps metrics keys to metrics data.
        optimizer : Optimizer
            Optimizer that optimizes the model.
        num_epochs : int
            Number of epochs of training.
        dataset_train : torch.utils.data.Dataset
            Dataset of training.
        dataset_valid : torch.utils.data.Dataset
            Dataset of validation/testing.
        batch_size : int
            Batch size.
        workers : int
            Number of workers used in data preprocessing.
        device : torch.device
            Device object. Either `torch.device("cuda")` or torch.device("cpu")`. When `None`, trainer will
            automatic detects GPU and selects GPU first.
        log_frequency : int
            Number of mini-batches to log metrics.
        callbacks : list of Callback
            Callbacks to plug into the trainer. See Callbacks.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model
        self.mutator = mutator
        self.loss = loss

        self.metrics = metrics
        self.optimizer = optimizer

        self.model.to(self.device)
        self.mutator.to(self.device)
        self.loss.to(self.device)

        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.batch_size = batch_size
        self.workers = workers
        self.log_frequency = log_frequency
        self.callbacks = callbacks if callbacks is not None else []
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def validate_one_epoch(self, epoch):
        pass

    def train(self, validate=True):
        for epoch in range(self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            _logger.info("Epoch %d Training", epoch)
            self.train_one_epoch(epoch)

            if validate:
                # validation
                _logger.info("Epoch %d Validating", epoch)
                self.validate_one_epoch(epoch)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

    def validate(self):
        self.validate_one_epoch(-1)

    def export(self, file):
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def checkpoint(self):
        raise NotImplementedError("Not implemented yet")
