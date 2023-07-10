# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import nni
from nni.compression.quantization import QATQuantizer
from nni.compression.utils import TorchEvaluator
from nni.common.types import SCHEDULER


torch.manual_seed(1024)
device = 'cuda:0'


MNIST(root='data/mnist', train=True, download=True)
MNIST(root='data/mnist', train=False, download=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST(root='data/mnist', train=True, transform=transform)
train_dataloader = DataLoader(mnist_train, batch_size=64)
mnist_test = MNIST(root='data/mnist', train=False, transform=transform)
test_dataloader = DataLoader(mnist_test, batch_size=1000)


class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        self.relu3 = torch.nn.ReLU6()
        self.max_pool1 = torch.nn.MaxPool2d(2, 2)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def training_step(batch, model):
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss: torch.Tensor = F.nll_loss(logits, y)
    return loss


def training_model(model: torch.nn.Module, optimizer: Optimizer, training_step: Callable, scheduler: SCHEDULER | None = None,
                   max_steps: int | None = None, max_epochs: int | None = None):
    model.train()
    max_epochs = max_epochs if max_epochs else 1 if max_steps is None else 100
    current_steps = 0

    # training
    for epoch in range(max_epochs):
        print(f'Epoch {epoch} start!')
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()
            current_steps += 1
            if max_steps and current_steps == max_steps:
                return
        if scheduler is not None:
            scheduler.step()


def evaluating_model(model: torch.nn.Module):
    model.eval()
    # testing
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y.view_as(preds)).sum().item()
    return correct / len(mnist_test)


model = NaiveModel().to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

start = time.time()
training_model(model, optimizer, training_step, None, None, 5)
print(f'pure training 5 epochs: {time.time() - start}s')
start = time.time()
acc = evaluating_model(model)
print(f'pure evaluating: {time.time() - start}s    Acc.: {acc}')

optimizer = nni.trace(SGD)(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
evaluator = TorchEvaluator(training_model, optimizer, training_step)  # type: ignore

config_list = [{
    'op_names': ['conv1', 'conv2', 'fc1', 'fc2'],
    'target_names': ['_input_', 'weight', '_output_'],
    'quant_dtype': 'int8',
    'quant_scheme': 'affine',
    'granularity': 'default',
},{
    'op_names': ['relu1', 'relu2', 'relu3'],
    'target_names': ['_output_'],
    'quant_dtype': 'int8',
    'quant_scheme': 'affine',
    'granularity': 'default',
}]

quantizer = QATQuantizer(model, config_list, evaluator, len(train_dataloader))
real_input = next(iter(train_dataloader))[0].to(device)
quantizer.track_forward(real_input)

start = time.time()
_, calibration_config = quantizer.compress(None, max_epochs=5)
print(f'pure training 5 epochs: {time.time() - start}s')

print(calibration_config)
start = time.time()
acc = evaluating_model(model)
print(f'quant evaluating: {time.time() - start}s    Acc.: {acc}')
