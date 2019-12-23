# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


class Callback:

    def __init__(self):
        self.model = None
        self.mutator = None
        self.trainer = None

    def build(self, model, mutator, trainer):
        self.model = model
        self.mutator = mutator
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, epoch):
        pass

    def on_batch_end(self, epoch):
        pass


class LRSchedulerCallback(Callback):
    def __init__(self, scheduler, mode="epoch"):
        super().__init__()
        assert mode == "epoch"
        self.scheduler = scheduler
        self.mode = mode

    def on_epoch_end(self, epoch):
        self.scheduler.step()


class ArchitectureCheckpoint(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch):
        dest_path = os.path.join(self.checkpoint_dir, "epoch_{}.json".format(epoch))
        _logger.info("Saving architecture to %s", dest_path)
        self.trainer.export(dest_path)


class ModelCheckpoint(Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch):
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        dest_path = os.path.join(self.checkpoint_dir, "epoch_{}.pth.tar".format(epoch))
        _logger.info("Saving model to %s", dest_path)
        torch.save(state_dict, dest_path)
