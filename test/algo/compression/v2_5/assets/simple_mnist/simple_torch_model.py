# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from ..device import device

from nni.common.types import SCHEDULER


class SimpleTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, groups=4)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(16, 32, 3)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.fc1 = torch.nn.Linear(32 * 24 * 24, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x)) + self.bn3(self.conv3(x))
        x = self.fc2(self.fc1(x.reshape(x.shape[0], -1)))
        return F.log_softmax(x, -1)


def training_step(batch: Tuple, model: Module, device: torch.device = device):
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss: torch.Tensor = F.nll_loss(logits, y)
    return loss


def training_model(model: Module, optimizer: Optimizer, training_step: Callable, scheduler: SCHEDULER = None,
                   max_steps: int | None = None, max_epochs: int | None = None, device: torch.device = device):
    model.train()

    # prepare data
    MNIST(root='data/mnist', train=True, download=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(root='data/mnist', train=True, transform=transform)
    train_dataloader = DataLoader(mnist_train, batch_size=32)

    max_epochs = max_epochs if max_epochs else 1
    max_steps = max_steps if max_steps else 50
    current_steps = 0

    # training
    for _ in range(max_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss: torch.Tensor = training_step(batch, model, device)
            loss.backward()
            optimizer.step()
            current_steps += 1
            if max_steps and current_steps == max_steps:
                return
        if scheduler is not None:
            scheduler.step()


def finetuning_model(model: Module):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    training_model(model, optimizer, F.nll_loss)


def evaluating_model(model: Module, device: torch.device = device):
    model.eval()

    # prepare data
    MNIST(root='data/mnist', train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_test = MNIST(root='data/mnist', train=False, transform=transform)
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
