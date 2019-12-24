# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator import SPOSSupernetTrainingMutator

logger = logging.getLogger(__name__)


class SPOSSupernetTrainer(Trainer):
    """
    This trainer trains a supernet that can be used for evolution search.
    """

    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, train_loader, valid_loader,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None):
        assert torch.cuda.is_available()
        super().__init__(model, mutator if mutator is not None else SPOSSupernetTrainingMutator(model),
                         loss, metrics, optimizer, num_epochs, None, None,
                         batch_size, workers, device, log_frequency, callbacks)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, (x, y) in enumerate(self.train_loader):
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
                logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                self.mutator.reset()
                logits = self.model(x)
                loss = self.loss(logits, y)
                metrics = self.metrics(logits, y)
                metrics["loss"] = loss.item()
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.valid_loader), meters)
