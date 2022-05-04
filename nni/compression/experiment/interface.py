# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .config.common import ComparableType


class NNILightningModule(Module):

    @abstractmethod
    def training_step(self, batch, batch_idx) -> Tensor:
        pass

    @abstractmethod
    def pruning_training_step(self, batch, batch_idx) -> ComparableType:
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx) -> ComparableType:
        pass

    @abstractmethod
    def criterion(self, output, target) -> Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        pass


class NNITrainer():
    def fit(self, model: NNILightningModule, train_dataloaders:Union[ DataLoader, Sequence[DataLoader]],
            val_dataloaders: Union[DataLoader, Sequence[DataLoader]]):
        if (isinstance(train_dataloaders, Sequence) and len(train_dataloaders) != 1) or \
            (isinstance(val_dataloaders, Sequence) and len(train_dataloaders) != 1):
            raise Error('Default fit() does not support sequence dataloader, please costomize NNITrainer.fit().')
        model.train()
        torch.set_grad_enabled(True)
        optimizer, lr_scheduler = model.configure_optimizers()

        for batch_idx, batch in enumerate(train_dataloaders):
            loss = model.training_step(batch, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    def validate(self, model: NNILightningModule, val_dataloader: DataLoader):
        pass


class NNIDataModule():
    @abstractmethod
    def train_dataloader(self) -> Sequence[DataLoader]:
        pass

    @abstractmethod
    def val_dataloader(self) -> Sequence[DataLoader]:
        pass
