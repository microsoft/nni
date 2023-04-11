"""
Quantization Quickstart
=======================

Here is a four-minute video to get you started with model quantization.

..  youtube:: MSfV7AyfiA4
    :align: center

Quantization reduces model size and speeds up inference time by reducing the number of bits required to represent weights or activations.

In NNI, both post-training quantization algorithms and quantization-aware training algorithms are supported.
Here we use `QATQuantizer` as an example to show the usage of quantization in NNI.
"""

# %%
# Preparation
# -----------
#
# In this tutorial, we use a simple model and pre-train on MNIST dataset.
# If you are familiar with defining a model and training in pytorch, you can skip directly to `Quantizing Model`_.

import functools
import time
from typing import Callable, Union, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch import Tensor

# define the model
class Mnist(torch.nn.Module):
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
        self.batchnorm1 = torch.nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.max_pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.max_pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Create training and evaluation dataloader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

MNIST(root='data/mnist', train=True, download=True)
MNIST(root='data/mnist', train=False, download=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST(root='data/mnist', train=True, transform=transform)
train_dataloader = DataLoader(mnist_train, batch_size=64)
mnist_test = MNIST(root='data/mnist', train=False, transform=transform)
test_dataloader = DataLoader(mnist_test, batch_size=1000)


# define training and evaluation functions
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def training_step(batch, model) -> Tensor:
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss: torch.Tensor = F.nll_loss(logits, y)
    return loss


def training_model(model: torch.nn.Module, optimizer: Union[Optimizer, List[Optimizer]], \
                   training_step, scheduler, max_steps: Union[int, None] = None, max_epochs: Union[int, None] = None):
    model.train()
    max_epochs = max_epochs or (40 if max_steps is None else 100)
    current_steps = 0
    best_acc = 0.0

    # training
    for epoch in range(max_epochs):
        print(f'Epoch {epoch} start!')
        for batch in train_dataloader:
            if isinstance(optimizer, Optimizer):
                optimizer.zero_grad()
            elif isinstance(optimizer, List) and all(isinstance(_, Optimizer) for _ in optimizer):
                for opt in optimizer:
                    opt.zero_grad()
            loss = training_step(batch, model)
            assert isinstance(loss, torch.Tensor)
            loss.backward()
            if isinstance(optimizer, Optimizer):
                optimizer.step()
            elif isinstance(optimizer, List) and all(isinstance(_, Optimizer) for _ in optimizer):
                for opt in optimizer:
                    opt.step()
            if isinstance(scheduler, _LRScheduler):
                scheduler.step()
            if isinstance(scheduler, List) and all(isinstance(_, _LRScheduler) for _ in scheduler):
                for sch in scheduler:
                    sch.step()
            current_steps += 1
            if max_steps and current_steps == max_steps:
                return

        acc = evaluating_model(model)
        best_acc = max(acc, best_acc)
        print(f"epoch={epoch}\tacc={acc}\tbest_acc={best_acc}")


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
    print(f'Accuracy: {100 * correct / len(mnist_test)}%)\n')

    return correct / len(mnist_test)


# pre-train and evaluate the model on MNIST dataset
model = Mnist().to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

start = time.time()
training_model(model, optimizer, training_step, None, None, 5)
print(f'pure training 5 epochs: {time.time() - start}s')
start = time.time()
acc = evaluating_model(model)
print(f'pure evaluating: {time.time() - start}s    Acc.: {acc}')


# %%
# Quantizing Model
# ----------------
#
# Initialize a `config_list`.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.