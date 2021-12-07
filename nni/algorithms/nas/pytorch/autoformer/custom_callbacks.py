# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os

import torch
import torch.nn as nn


from .utils import save_on_master

_logger = logging.getLogger(__name__)


class Callback:
    """
    Callback provides an easy way to react to events like begin/end of epochs.
    """

    def __init__(self):
        self.model = None
        self.mutator = None
        self.trainer = None

    def build(self, model, mutator, trainer):
        """
        Callback needs to be built with model, mutator, trainer, to get updates from them.

        Parameters
        ----------
        model : nn.Module
            Model to be trained.
        mutator : nn.Module
            Mutator that mutates the model.
        trainer : BaseTrainer
            Trainer that is to call the callback.
        """
        self.model = model
        self.mutator = mutator
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        """
        Implement this to do something at the begin of epoch.

        Parameters
        ----------
        epoch : int
            Epoch number, starting from 0.
        """
        pass

    def on_epoch_end(self, epoch):
        """
        Implement this to do something at the end of epoch.

        Parameters
        ----------
        epoch : int
            Epoch number, starting from 0.
        """
        pass

    def on_batch_begin(self, epoch):
        pass

    def on_batch_end(self, epoch):
        pass


class LRSchedulerCallback(Callback):
    """
    Calls scheduler on every epoch ends.

    Parameters
    ----------
    scheduler : LRScheduler
        Scheduler to be called.
    """
    def __init__(self, scheduler, mode="epoch"):
        super().__init__()
        assert mode == "epoch"
        self.scheduler = scheduler
        self.mode = mode

    def on_epoch_end(self, epoch):
        """
        Call ``self.scheduler.step()`` on epoch end.
        """
        self.scheduler.step(epoch)


class SaveCheckpointCallback(Callback):
    """
    Calls ``trainer.export()`` on every epoch ends.

    Parameters
    ----------
    checkpoint_dir : str
        Location to save checkpoints.
    """
    def __init__(self, output_dir, optimizer, lr_scheduler, loss_scaler, args):
        super().__init__()
        self.output_dir = output_dir
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_scaler = loss_scaler
        self.args = args

    def on_epoch_end(self, epoch):
        """
        Dump to ``/checkpoint_dir/epoch_{number}.pth.tar`` on every epoch end.
        ``DataParallel`` object will have their inside modules exported.
        """
        # if isinstance(self.model, nn.DataParallel):
        #     state_dict = self.model.module.state_dict()
        # else:
        #     state_dict = self.model.state_dict()
        # dest_path = os.path.join(self.checkpoint_dir, "epoch_{}.pth.tar".format(epoch))
        # _logger.info("Saving model to %s", dest_path)
        # torch.save(state_dict, dest_path)


        checkpoint_paths = [self.output_dir / 'checkpoint.pth']
        if hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        for checkpoint_path in checkpoint_paths:
            save_on_master({
                'model': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': self.loss_scaler.state_dict(),
                'args': self.args,
            }, checkpoint_path)