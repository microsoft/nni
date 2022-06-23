# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path
import pickle

import pytest

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
pl.seed_everything(10)

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split, DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision import transforms

import nni
from nni.algorithms.compression.v2.pytorch.utils.evaluator import (
    Evaluator,
    LegacyEvaluator,
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


def test_lightning_evaluator():
    pl_model = SimpleLightningModel()
    pl_trainer = nni.trace(pl.Trainer)(
        max_epochs=1,
        max_steps=10,
        logger=TensorBoardLogger(Path(__file__).parent / 'lightning_logs', name="resnet"),
    )
    pl_data = MNISTDataModule(data_dir=Path(__file__).parent / 'data')
    evaluator = LightningEvaluator(pl_trainer, pl_data)
    evaluator._init_optimizer_helpers(pl_model)

    # test evaluator reuse
    new_pl_model = SimpleLightningModel()
    evaluator.bind_model(new_pl_model)

    tensor_hook = TensorHook(new_pl_model.model.conv1.weight, 'model.conv1.weight', tensor_hook_factory)
    forward_hook = ForwardHook(new_pl_model.model.conv1, 'model.conv1', forward_hook_factory)
    backward_hook = BackwardHook(new_pl_model.model.conv1, 'model.conv1', backward_hook_factory)

    # test train with patch & hook
    reset_flags()
    evaluator.patch_loss(loss_patch)
    evaluator.patch_optimizer_step([optimizer_before_step_patch], [optimizer_after_step_patch])
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.train(max_steps=1)
    assert_flags()
    # `len(forward_hook.buffer) == 3` is because lightning will dry run two steps at first for sanity check
    # pl_trainer.num_sanity_val_steps = 0 can drop sanity check
    assert all([len(tensor_hook.buffer) == 1, len(forward_hook.buffer) == 3, len(backward_hook.buffer) == 1])

    # test finetune with patch & hook
    reset_flags()
    evaluator.remove_all_hooks()
    evaluator.register_hooks([tensor_hook, forward_hook, backward_hook])

    evaluator.finetune()
    assert_flags()
    assert all([len(tensor_hook.buffer) == 10, len(forward_hook.buffer) == 12, len(backward_hook.buffer) == 10])


def test_legacy_evaluator():
    pass
