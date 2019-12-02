# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator import SPOSSupernetTrainingMutator

logger = logging.getLogger(__name__)


class SPOSSupernetTrainer(Trainer):
    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None):
        super().__init__(model, mutator if mutator is not None else SPOSSupernetTrainingMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                        batch_size=batch_size,
                                                        num_workers=workers)

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss(x, y)
            loss.backward()
            self.optimizer.step()

            metrics = self.metrics(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

    def validate_one_epoch(self, epoch):
        self.model.validate()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                metrics = self.metrics(logits, y)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.valid_loader), meters)
