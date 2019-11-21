import json
import logging
from abc import abstractmethod

import torch

from .base_trainer import BaseTrainer

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


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
            _logger.info("Epoch {} Training".format(epoch))
            self.train_one_epoch(epoch)

            if validate:
                # validation
                _logger.info("Epoch {} Validating".format(epoch))
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
