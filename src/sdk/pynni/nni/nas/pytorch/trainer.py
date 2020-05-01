# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import time
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
    """
    A trainer with some helper functions implemented. To implement a new trainer,
    users need to implement :meth:`train_one_epoch`, :meth:`validate_one_epoch` and :meth:`checkpoint`.

    Parameters
    ----------
    model : nn.Module
        Model with mutables.
    mutator : BaseMutator
        A mutator object that has been initialized with the model.
    loss : callable
        Called with logits and targets. Returns a loss tensor.
        See `PyTorch loss functions`_ for examples.
    metrics : callable
        Called with logits and targets. Returns a dict that maps metrics keys to metrics data. For example,

        .. code-block:: python

            def metrics_fn(output, target):
                return {"acc1": accuracy(output, target, topk=1), "acc5": accuracy(output, target, topk=5)}

    optimizer : Optimizer
        Optimizer that optimizes the model.
    num_epochs : int
        Number of epochs of training.
    dataset_train : torch.utils.data.Dataset
        Dataset of training. If not otherwise specified, ``dataset_train`` and ``dataset_valid`` should be standard
        PyTorch Dataset. See `torch.utils.data`_ for examples.
    dataset_valid : torch.utils.data.Dataset
        Dataset of validation/testing.
    batch_size : int
        Batch size.
    workers : int
        Number of workers used in data preprocessing.
    device : torch.device
        Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
        automatic detects GPU and selects GPU first.
    log_frequency : int
        Number of mini-batches to log metrics.
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.


    .. _`PyTorch loss functions`: https://pytorch.org/docs/stable/nn.html#loss-functions
    .. _`torch.utils.data`: https://pytorch.org/docs/stable/data.html
    """
    def __init__(self, model, mutator, loss, metrics, optimizer, num_epochs,
                 dataset_train, dataset_valid, batch_size, workers, device, log_frequency, callbacks):
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
        self.log_dir = os.path.join("logs", str(time.time()))
        os.makedirs(self.log_dir, exist_ok=True)
        self.status_writer = open(os.path.join(self.log_dir, "log"), "w")
        self.callbacks = callbacks if callbacks is not None else []
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)

    @abstractmethod
    def train_one_epoch(self, epoch):
        """
        Train one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number starting from 0.
        """
        pass

    @abstractmethod
    def validate_one_epoch(self, epoch):
        """
        Validate one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number starting from 0.
        """
        pass

    def train(self, validate=True):
        """
        Train ``num_epochs``.
        Trigger callbacks at the start and the end of each epoch.

        Parameters
        ----------
        validate : bool
            If ``true``, will do validation every epoch.
        """
        for epoch in range(self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            _logger.info("Epoch %d Training", epoch + 1)
            self.train_one_epoch(epoch)

            if validate:
                # validation
                _logger.info("Epoch %d Validating", epoch + 1)
                self.validate_one_epoch(epoch)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

    def validate(self):
        """
        Do one validation.
        """
        self.validate_one_epoch(-1)

    def export(self, file):
        """
        Call ``mutator.export()`` and dump the architecture to ``file``.

        Parameters
        ----------
        file : str
            A file path. Expected to be a JSON.
        """
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def checkpoint(self):
        """
        Return trainer checkpoint.
        """
        raise NotImplementedError("Not implemented yet")

    def enable_visualization(self):
        """
        Enable visualization. Write graph and training log to folder ``logs/<timestamp>``.
        """
        sample = None
        for x, _ in self.train_loader:
            sample = x.to(self.device)[:2]
            break
        if sample is None:
            _logger.warning("Sample is %s.", sample)
        _logger.info("Creating graph json, writing to %s. Visualization enabled.", self.log_dir)
        with open(os.path.join(self.log_dir, "graph.json"), "w") as f:
            json.dump(self.mutator.graph(sample), f)
        self.visualization_enabled = True

    def _write_graph_status(self):
        if hasattr(self, "visualization_enabled") and self.visualization_enabled:
            print(json.dumps(self.mutator.status()), file=self.status_writer, flush=True)
