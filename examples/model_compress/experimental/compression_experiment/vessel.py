# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms

import nni


@nni.trace
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
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

_use_cuda = True
device = torch.device("cuda" if _use_cuda else "cpu")

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

_train_loader = None
_test_loader = None

def trainer(model, optimizer, criterion):
    global _train_loader
    if _train_loader is None:
        dataset = datasets.MNIST('./data', train=True, download=True, transform=_transform)
        _train_loader = torch.utils.data.DataLoader(dataset, **_train_kwargs)
    model.train()
    for data, target in _train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluator(model):
    global _test_loader
    if _test_loader is None:
        dataset = datasets.MNIST('./data', train=False, transform=_transform, download=True)
        _test_loader = torch.utils.data.DataLoader(dataset, **_test_kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(_test_loader.dataset)
    acc = 100 * correct / len(_test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(_test_loader.dataset), acc))
    return acc

criterion = F.nll_loss

def finetuner(model: nn.Module):
    optimizer = Adam(model.parameters())
    for i in range(3):
        trainer(model, optimizer, criterion)
