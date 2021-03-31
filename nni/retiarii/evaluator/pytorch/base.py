# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# This file is deprecated.

import abc
from typing import Any, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni

class BaseTrainer(abc.ABC):
    @abc.abstractmethod
    def fit(self) -> None:
        pass


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
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
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
        super().__init__()
        self._use_cuda = torch.cuda.is_available()
        self.model = model
        if self._use_cuda:
            self.model.cuda()
        self._loss_fn = nn.CrossEntropyLoss()
        self._train_dataset = getattr(datasets, dataset_cls)(train=True, transform=get_default_transform(dataset_cls),
                                                             **(dataset_kwargs or {}))
        self._val_dataset = getattr(datasets, dataset_cls)(train=False, transform=get_default_transform(dataset_cls),
                                                           **(dataset_kwargs or {}))
        self._optimizer = getattr(torch.optim, optimizer_cls)(model.parameters(), **(optimizer_kwargs or {}))
        self._trainer_kwargs = trainer_kwargs or {'max_epochs': 10}

        self._train_dataloader = DataLoader(self._train_dataset, **(dataloader_kwargs or {}))
        self._val_dataloader = DataLoader(self._val_dataset, **(dataloader_kwargs or {}))

    def _accuracy(self, input, target):  # pylint: disable=redefined-builtin
        _, predict = torch.max(input.data, 1)
        correct = predict.eq(target.data).cpu().sum().item()
        return correct / input.size(0)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = self.training_step_before_model(batch, batch_idx)
        y_hat = self.model(x)
        return self.training_step_after_model(x, y, y_hat)

    def training_step_before_model(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        if self._use_cuda:
            x, y = x.cuda(torch.device('cuda:0')), y.cuda(torch.device('cuda:0'))
        return x, y

    def training_step_after_model(self, x, y, y_hat):
        loss = self._loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = self.validation_step_before_model(batch, batch_idx)
        y_hat = self.model(x)
        return self.validation_step_after_model(x, y, y_hat)

    def validation_step_before_model(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        if self._use_cuda:
            x, y = x.cuda(), y.cuda()
        return x, y

    def validation_step_after_model(self, x, y, y_hat):
        acc = self._accuracy(y_hat, y)
        return {'val_acc': acc}

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # We might need dict metrics in future?
        avg_acc = np.mean([x['val_acc'] for x in outputs]).item()
        nni.report_intermediate_result(avg_acc)
        return {'val_acc': avg_acc}

    def _validate(self):
        validation_outputs = []
        for i, batch in enumerate(self._val_dataloader):
            validation_outputs.append(self.validation_step(batch, i))
        return self.validation_epoch_end(validation_outputs)

    def _train(self):
        for i, batch in enumerate(self._train_dataloader):
            self._optimizer.zero_grad()
            loss = self.training_step(batch, i)
            loss.backward()
            self._optimizer.step()

    def fit(self) -> None:
        for _ in range(self._trainer_kwargs['max_epochs']):
            self._train()
            self._validate()
        # assuming val_acc here
        nni.report_final_result(self._validate()['val_acc'])


class PyTorchMultiModelTrainer(BaseTrainer):
    def __init__(self, multi_model, kwargs=[]):
        self.multi_model = multi_model
        self.kwargs = kwargs
        self._train_dataloaders = []
        self._train_datasets = []
        self._val_dataloaders = []
        self._val_datasets = []
        self._optimizers = []
        self._trainers = []
        self._loss_fn = nn.CrossEntropyLoss()
        self.max_steps = self.kwargs['max_steps'] if 'makx_steps' in self.kwargs else None
        self.n_model = len(self.kwargs['model_kwargs'])

        for m in self.kwargs['model_kwargs']:
            if m['use_input']:
                dataset_cls = m['dataset_cls']
                dataset_kwargs = m['dataset_kwargs']
                dataloader_kwargs = m['dataloader_kwargs']
                train_dataset = getattr(datasets, dataset_cls)(train=True, transform=get_default_transform(dataset_cls),
                                                               **(dataset_kwargs or {}))
                val_dataset = getattr(datasets, dataset_cls)(train=False, transform=get_default_transform(dataset_cls),
                                                             **(dataset_kwargs or {}))
                train_dataloader = DataLoader(train_dataset, **(dataloader_kwargs or {}))
                val_dataloader = DataLoader(val_dataset, **(dataloader_kwargs or {}))
                self._train_datasets.append(train_dataset)
                self._train_dataloaders.append(train_dataloader)

                self._val_datasets.append(val_dataset)
                self._val_dataloaders.append(val_dataloader)

            if m['use_output']:
                optimizer_cls = m['optimizer_cls']
                optimizer_kwargs = m['optimizer_kwargs']
                m_header = f"M_{m['model_id']}"
                one_model_params = []
                for name, param in multi_model.named_parameters():
                    name_prefix = '_'.join(name.split('_')[:2])
                    if m_header == name_prefix:
                        one_model_params.append(param)

                optimizer = getattr(torch.optim, optimizer_cls)(one_model_params, **(optimizer_kwargs or {}))
                self._optimizers.append(optimizer)

    def fit(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        max_epochs = max([x['trainer_kwargs']['max_epochs'] for x in self.kwargs['model_kwargs']])
        for _ in range(max_epochs):
            self._train()
            self._validate()
        nni.report_final_result(self._validate())

    def _train(self):
        for batch_idx, multi_model_batch in enumerate(zip(*self._train_dataloaders)):
            for opt in self._optimizers:
                opt.zero_grad()
            xs = []
            ys = []
            for idx, batch in enumerate(multi_model_batch):
                x, y = self.training_step_before_model(batch, batch_idx, f'cuda:{idx}')
                xs.append(x)
                ys.append(y)

            y_hats = self.multi_model(*xs)
            if len(ys) != len(xs):
                raise ValueError('len(ys) should be equal to len(xs)')
            losses = []
            report_loss = {}
            for output_idx, yhat in enumerate(y_hats):
                if len(ys) == len(y_hats):
                    loss = self.training_step_after_model(xs[output_idx], ys[output_idx], yhat)
                elif len(ys) == 1:
                    loss = self.training_step_after_model(xs[0], ys[0].to(yhat.get_device()), yhat)
                else:
                    raise ValueError('len(ys) should be either 1 or len(y_hats)')
                losses.append(loss.to("cuda:0"))
                report_loss[self.kwargs['model_kwargs'][output_idx]['model_id']] = loss.item()
            summed_loss = sum(losses)
            summed_loss.backward()
            for opt in self._optimizers:
                opt.step()
            if self.max_steps and batch_idx >= self.max_steps:
                return

    def training_step_before_model(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, device=None):
        x, y = batch
        if device:
            x, y = x.cuda(torch.device(device)), y.cuda(torch.device(device))
        return x, y

    def training_step_after_model(self, x, y, y_hat):
        loss = self._loss_fn(y_hat, y)
        return loss

    def _validate(self):
        all_val_outputs = {idx: [] for idx in range(self.n_model)}
        for batch_idx, multi_model_batch in enumerate(zip(*self._val_dataloaders)):
            xs = []
            ys = []
            for idx, batch in enumerate(multi_model_batch):
                x, y = self.training_step_before_model(batch, batch_idx, f'cuda:{idx}')
                xs.append(x)
                ys.append(y)
            if len(ys) != len(xs):
                raise ValueError('len(ys) should be equal to len(xs)')

            y_hats = self.multi_model(*xs)

            for output_idx, yhat in enumerate(y_hats):
                if len(ys) == len(y_hats):
                    acc = self.validation_step_after_model(xs[output_idx], ys[output_idx], yhat)
                elif len(ys) == 1:
                    acc = self.validation_step_after_model(xs[0], ys[0].to(yhat.get_device()), yhat)
                else:
                    raise ValueError('len(ys) should be either 1 or len(y_hats)')
                all_val_outputs[output_idx].append(acc)

        report_acc = {}
        for idx in all_val_outputs:
            avg_acc = np.mean([x['val_acc'] for x in all_val_outputs[idx]]).item()
            report_acc[self.kwargs['model_kwargs'][idx]['model_id']] = avg_acc
        nni.report_intermediate_result(report_acc)
        return report_acc

    def validation_step_before_model(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, device=None):
        x, y = batch
        if device:
            x, y = x.cuda(torch.device(device)), y.cuda(torch.device(device))
        return x, y

    def validation_step_after_model(self, x, y, y_hat):
        acc = self._accuracy(y_hat, y)
        return {'val_acc': acc}

    def _accuracy(self, input, target):  # pylint: disable=redefined-builtin
        _, predict = torch.max(input.data, 1)
        correct = predict.eq(target.data).cpu().sum().item()
        return correct / input.size(0)
