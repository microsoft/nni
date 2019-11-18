from abc import abstractmethod

import torch

from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, optimizer, num_epochs,
                 dataset_train, dataset_valid, batch_size, workers, device, log_frequency,
                 mutator, callbacks):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.mutator = mutator

        self.model.to(self.device)
        self.loss.to(self.device)
        self.mutator.to(self.device)

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

    def _train(self, validate):
        for epoch in range(self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            print("Epoch {} Training".format(epoch))
            self.train_one_epoch(epoch)

            if validate:
                # validation
                print("Epoch {} Validating".format(epoch))
                self.validate_one_epoch(epoch)

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

    def train_and_validate(self):
        self._train(True)

    def train(self):
        self._train(False)

    def validate(self):
        self.validate_one_epoch(-1)
