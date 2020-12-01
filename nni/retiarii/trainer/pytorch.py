import abc
from typing import *

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni

from .interface import BaseTrainer


def get_default_transform(dataset: str) -> Any:
    """
    To get a default transformation of image for a specific dataset.
    This is needed because transform objects can not be directly passed as arguments.

    Parameters
    ----------
    dataset : str
        Dataset class name.

    Returns
    -------
    transform object
    """
    if dataset == 'MNIST':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if dataset == 'CIFAR10':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # unsupported dataset, return None
    return None


class PyTorchImageClassificationTrainer(BaseTrainer):
    """
    Image classification trainer for PyTorch.

    A model, along with corresponding dataset, optimizer config is used to initialize the trainer.
    The trainer will run for a fixed number of epochs (by default 10), and report the final result.

    TODO
    Support scheduler, validate every n epochs, train/valid dataset

    Limitation induced by NNI: kwargs must be serializable to put into a JSON packed in parameters.
    """

    def __init__(self, model,
                 dataset_cls='MNIST', dataset_kwargs=None, dataloader_kwargs=None,
                 optimizer_cls='SGD', optimizer_kwargs=None, trainer_kwargs=None):
        """Initialization of image classification trainer.

        Parameters
        ----------
        model : nn.Module
            Model to train.
        dataset_cls : str, optional
            Dataset class name that is available in ``torchvision.datasets``, by default 'MNIST'
        dataset_kwargs : dict, optional
            Keyword arguments passed to initialization of dataset class, by default None
        dataset_kwargs : dict, optional
            Keyword arguments passed to ``torch.utils.data.DataLoader``, by default None
        optimizer_cls : str, optional
            Optimizer class name that is available in ``torch.optim``, by default 'SGD'
        optimizer_kwargs : dict, optional
            Keyword arguments passed to initialization of optimizer class, by default None
        trainer_kwargs: dict, optional
            Keyword arguments passed to trainer. Will be passed to Trainer class in future. Currently,
            only the key ``max_epochs`` is useful.
        """
        super(PyTorchImageClassificationTrainer, self).__init__(model,
                 dataset_cls, dataset_kwargs, dataloader_kwargs,
                 optimizer_cls, optimizer_kwargs, trainer_kwargs)
        self._use_cuda = torch.cuda.is_available()
        self.model = model
        if self._use_cuda:
            self.model.cuda()
        self._loss_fn = nn.CrossEntropyLoss()
        self._dataset = getattr(datasets, dataset_cls)(transform=get_default_transform(dataset_cls),
                                                       **(dataset_kwargs or {}))
        self._optimizer = getattr(torch.optim, optimizer_cls)(model.parameters(), **(optimizer_kwargs or {}))
        self._trainer_kwargs = trainer_kwargs or {'max_epochs': 10}

        # TODO: we will need at least two (maybe three) data loaders in future.
        self._dataloader = DataLoader(self._dataset, **(dataloader_kwargs or {}))

    def _accuracy(self, input, target):
        _, predict = torch.max(input.data, 1)
        correct = predict.eq(target.data).cpu().sum().item()
        return correct / input.size(0)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        if self._use_cuda:
            x, y = x.cuda(), y.cuda()
        y_hat = self.model(x)
        loss = self._loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        if self._use_cuda:
            x, y = x.cuda(), y.cuda()
        y_hat = self.model(x)
        acc = self._accuracy(y_hat, y)
        return {'val_acc': acc}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # We might need dict metrics in future?
        avg_acc = np.mean([x['val_acc'] for x in outputs]).item()
        nni.report_intermediate_result(avg_acc)
        return {'val_acc': avg_acc}

    def _validate(self):
        validation_outputs = []
        for i, batch in enumerate(self._dataloader):
            validation_outputs.append(self.validation_step(batch, i))
        return self.validation_epoch_end(validation_outputs)

    def _train(self):
        for i, batch in enumerate(self._dataloader):
            self.training_step(batch, i)

    def fit(self) -> None:
        for _ in range(self._trainer_kwargs['max_epochs']):
            self._train()
        nni.report_final_result(self._validate()['val_acc'])  # assuming val_acc here
