# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from nni.algorithms.compression.pytorch.auto_compress import AbstractBasket


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

_use_cuda = torch.cuda.is_available()

_train_kwargs = {'batch_size': 64}
_test_kwargs = {'batch_size': 1000}
if _use_cuda:
    _cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    _train_kwargs.update(_cuda_kwargs)
    _test_kwargs.update(_cuda_kwargs)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

_dataset1 = datasets.MNIST('./data', train=True, download=True, transform=_transform)
_dataset2 = datasets.MNIST('./data', train=False, transform=_transform)
_train_loader = torch.utils.data.DataLoader(_dataset1, **_train_kwargs)
_test_loader = torch.utils.data.DataLoader(_dataset2, **_test_kwargs)

_device = torch.device("cuda" if _use_cuda else "cpu")
_epoch = 2

def _train(model, optimizer):
    model.train()
    for data, target in _train_loader:
        data, target = data.to(_device), target.to(_device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def _test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(_test_loader.dataset)
    acc = 100 * correct / len(_test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(_test_loader.dataset), acc))
    return acc

_model = LeNet().to(_device)

_pre_train_optimizer = optim.Adadelta(_model.parameters(), lr=1)
_scheduler = StepLR(_pre_train_optimizer, step_size=1, gamma=0.7)
for _ in range(_epoch):
    _train(_model, _pre_train_optimizer)
    _test(_model)
    _scheduler.step()

class Basket(AbstractBasket):
    @classmethod
    def model(cls) -> nn.Module:
        return _model

    @classmethod
    def optimizer(cls) -> torch.optim.Optimizer:
        return torch.optim.SGD(_model.parameters(), lr=0.01)

    @classmethod
    def evaluator(cls) -> Callable[[nn.Module], float]:
        return _test

    @classmethod
    def finetune_trainer(cls, compressor_type: str, algorithm_name: str) -> Optional[Callable[[nn.Module, optim.Optimizer], None]]:
        def _trainer(model, optimizer):
            for _ in range(_epoch):
                _train(model, optimizer)
        return _trainer
