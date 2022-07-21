# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path
from typing import Callable
import pytest

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.utils.data import random_split, DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision import transforms

import nni
from nni.algorithms.compression.v2.pytorch.utils.evaluator import (
    TorchEvaluator,
    LightningEvaluator,
    TensorHook,
    ForwardHook,
    BackwardHook,
)


class SimpleTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, groups=4)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(16, 8, 3)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.fc1 = torch.nn.Linear(8 * 24 * 24, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x)) + self.bn3(self.conv3(x))
        x = self.fc2(self.fc1(x.reshape(x.shape[0], -1)))
        return F.log_softmax(x, -1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_model(model: Module, optimizer: Optimizer, criterion: Callable, scheduler: _LRScheduler,
                   max_steps: int | None = None, max_epochs: int | None = None):
    model.train()

    # prepare data
    data_dir = Path(__file__).parent / 'data'
    MNIST(data_dir, train=True, download=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(data_dir, train=True, transform=transform)
    train_dataloader = DataLoader(mnist_train, batch_size=32)

    max_epochs = max_epochs if max_epochs else 1
    max_steps = max_steps if max_steps else 10
    current_steps = 0

    # training
    for _ in range(max_epochs):
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss: torch.Tensor = criterion(logits, y)
            loss.backward()
            optimizer.step()
            current_steps += 1
            if max_steps and current_steps == max_steps:
                return
        scheduler.step()


def evaluating_model(model: Module):
    model.eval()

    # prepare data
    data_dir = Path(__file__).parent / 'data'
    MNIST(data_dir, train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_test = MNIST(data_dir, train=False, transform=transform)
    test_dataloader = DataLoader(mnist_test, batch_size=32)

    # testing
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y.view_as(preds)).sum().item()
    return correct / len(mnist_test)


class SimpleLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleTorchModel()
        self.count = 0

    def forward(self, x):
        print(self.count)
        self.count += 1
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = nni.trace(torch.optim.SGD)(
            self.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": nni.trace(ExponentialLR)(
                optimizer,
                0.1,
            ),
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


optimizer_before_step_flag = False
optimizer_after_step_flag = False
loss_flag = False

def optimizer_before_step_patch():
    global optimizer_before_step_flag
    optimizer_before_step_flag = True

def optimizer_after_step_patch():
    global optimizer_after_step_flag
    optimizer_after_step_flag = True

def loss_patch(t: torch.Tensor):
    global loss_flag
    loss_flag = True
    return t

def tensor_hook_factory(buffer: list):
    def hook_func(t: torch.Tensor):
        buffer.append(True)
    return hook_func

def forward_hook_factory(buffer: list):
    def hook_func(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        buffer.append(True)
    return hook_func

def backward_hook_factory(buffer: list):
    def hook_func(module: torch.nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        buffer.append(True)
    return hook_func

def reset_flags():
    global optimizer_before_step_flag, optimizer_after_step_flag, loss_flag
    optimizer_before_step_flag = False
    optimizer_after_step_flag = False
    loss_flag = False

def assert_flags():
    global optimizer_before_step_flag, optimizer_after_step_flag, loss_flag
    assert optimizer_before_step_flag, 'Evaluator patch optimizer before step failed.'
    assert optimizer_after_step_flag, 'Evaluator patch optimizer after step failed.'
    assert loss_flag, 'Evaluator patch loss failed.'


def create_lighting_evaluator():
    pl_model = SimpleLightningModel()
    pl_trainer = nni.trace(pl.Trainer)(
        max_epochs=1,
        max_steps=10,
        logger=TensorBoardLogger(Path(__file__).parent / 'lightning_logs', name="resnet"),
    )
    pl_trainer.num_sanity_val_steps = 0
    pl_data = nni.trace(MNISTDataModule)(data_dir=Path(__file__).parent / 'data')
    evaluator = LightningEvaluator(pl_trainer, pl_data)
    evaluator._init_optimizer_helpers(pl_model)
    return evaluator


def create_pytorch_evaluator():
    model = SimpleTorchModel()
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = nni.trace(ExponentialLR)(optimizer, 0.1)
    evaluator = TorchEvaluator(training_model, optimizer, F.nll_loss, lr_scheduler, evaluating_func=evaluating_model)
    evaluator._init_optimizer_helpers(model)
    return evaluator


@pytest.mark.parametrize("evaluator_type", ['lightning', 'pytorch'])
def test_evaluator(evaluator_type: str):
    if evaluator_type == 'lightning':
        evaluator = create_lighting_evaluator()
        model = SimpleLightningModel()
        evaluator.bind_model(model)
        tensor_hook = TensorHook(model.model.conv1.weight, 'model.conv1.weight', tensor_hook_factory)
        forward_hook = ForwardHook(model.model.conv1, 'model.conv1', forward_hook_factory)
        backward_hook = BackwardHook(model.model.conv1, 'model.conv1', backward_hook_factory)
    elif evaluator_type == 'pytorch':
        evaluator = create_pytorch_evaluator()
        model = SimpleTorchModel().to(device)
        evaluator.bind_model(model)
        tensor_hook = TensorHook(model.conv1.weight, 'conv1.weight', tensor_hook_factory)
        forward_hook = ForwardHook(model.conv1, 'conv1', forward_hook_factory)
        backward_hook = BackwardHook(model.conv1, 'conv1', backward_hook_factory)
    else:
        raise ValueError(f'wrong evaluator_type: {evaluator_type}')

    # test train with patch & hook
    reset_flags()
    evaluator.patch_loss(loss_patch)
    evaluator.patch_optimizer_step([optimizer_before_step_patch], [optimizer_after_step_patch])
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.train(max_steps=1)
    assert_flags()
    assert all([len(hook.buffer) == 1 for hook in [tensor_hook, forward_hook, backward_hook]])

    # test finetune with patch & hook
    reset_flags()
    evaluator.remove_all_hooks()
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.finetune()
    assert_flags()
    assert all([len(hook.buffer) == 10 for hook in [tensor_hook, forward_hook, backward_hook]])
