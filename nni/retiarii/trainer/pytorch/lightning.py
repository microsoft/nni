import abc
from typing import Union, Optional, List

import nni
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...graph import TrainingConfig
from ...utils import blackbox_module


__all__ = ['LightningModule', 'Trainer', 'DataLoader', 'Lightning', 'SupervisedLearning']


@abc.ABC
class LightningModule(pl.LightningModule):
    @abc.abstractmethod
    def set_model(self, model_cls):
        pass


Trainer = blackbox_module(pl.Trainer)
DataLoader = blackbox_module(DataLoader)


class Lightning(TrainingConfig):
    """
    Delegate the whole training to PyTorch Lightning.

    Since the arguments passed to the initialization needs to be serialized, ``LightningModule``, ``Trainer`` or
    ``DataLoader`` in this file should be used. Another option is to hide dataloader in the Lightning module, in
    which case, dataloaders are not required for this class to work.

    Parameters
    ----------
    lightning_module : LightningModule
        Lightning module that defines the training logic.
    trainer : Trainer
        Lightning trainer that handles the training.
    train_dataloders : DataLoader
        Used in ``trainer.fit()``. A PyTorch DataLoader with training samples.
        If the ``lightning_module`` has a predefined train_dataloader method this will be skipped.
    val_dataloaders : DataLoader or List of DataLoader
        Used in ``trainer.fit()``. Either a single PyTorch Dataloader or a list of them, specifying validation samples.
        If the ``lightning_module`` has a predefined val_dataloaders method this will be skipped.
    """

    def __init__(self, lightning_module: LightningModule, trainer: Trainer,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloaders: Union[DataLoader, List[DataLoader], None] = None):
        assert isinstance(lightning_module, LightningModule), f'Lightning module must be an instance of {__name__}.LightningModule.'
        assert isinstance(trainer, Trainer), f'Trainer must be imported from {__name__}.'
        self.module = lightning_module
        self.trainer = trainer
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders

    @staticmethod
    def _load(ir):
        return Lightning(ir['module'], ir['trainer'], ir['train_dataloader'], ir['val_dataloaders'])

    def _dump(self):
        return {
            'module': self.module,
            'trainer': self.trainer,
            'train_dataloader': self.train_dataloader,
            'val_dataloaders': self.val_dataloaders
        }

    def _execute(self, model_cls):
        self.module.set_model(model_cls)
        return self.trainer.fit(self.module, self.train_dataloader, self.val_dataloaders)

    def __eq__(self, other):
        return self.function == other.function and self.arguments == other.arguments


### The following are some commonly used Lightning modules ###

@blackbox_module
class SupervisedLearning(LightningModule):
    def __init__(self, criterion=nn.CrossEntropyLoss,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.,
                 optimizer: optim.Optimizer = optim.Adam):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = criterion()
        self.optimizer = optimizer

    def set_model(self, model_cls):
        self.model = model_cls()

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # TODO: report accuracy
        return {'val_loss': self.criterion(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss}
        progress_bar_metrics = tensorboard_logs
        nni.report_intermediate_result(val_loss)
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': self.criterion(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_mse_loss': test_loss}
        progress_bar_metrics = tensorboard_logs
        return {'test_loss': test_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)
