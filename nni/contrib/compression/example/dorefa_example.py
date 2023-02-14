# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from typing import Callable, Union, List

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import nni
from nni.contrib.compression.quantization import DoReFaQuantizer
from nni.contrib.compression.utils import TorchEvaluator

torch.manual_seed(0)
device = 'cuda'

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


def training_model(train_dataloader: DataLoader, test_dataloader: DataLoader, model: torch.nn.Module, optimizer: Optimizer, \
                   training_step: Callable, scheduler: Union[_LRScheduler, None] = None,
                   max_steps: Union[int, None] = None, max_epochs: Union[int, None] = None):
    model.train()
    max_epochs = max_epochs or (10 if max_steps is None else 100)
    current_steps = 0
    best_acc = 0.0

    # training
    for epoch in range(max_epochs):
        print(f'Epoch {epoch} start!')
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, _LRScheduler) or (isinstance(scheduler, List) and len(scheduler) > 0):
                scheduler.step()
            current_steps += 1
            if max_steps and current_steps == max_steps:
                return

        acc = evaluating_model(model, test_dataloader)
        best_acc = max(acc, best_acc)
        print(f"epoch={epoch}\tacc={acc}\tbest_acc={best_acc}")


def evaluating_model(model: torch.nn.Module, test_dataloader: DataLoader):
    model.eval()
    # testing
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y.view_as(preds)).sum().item()

    print(f'Accuracy: {100 * correct / len(test_dataloader.dataset)}%)\n')

    return correct / len(test_dataloader.dataset)


def main():

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=trans),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=trans),
        batch_size=1000, shuffle=True)

    model = NaiveModel()
    model = model.to(device)
    configure_list = [{
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
    training_func = functools.partial(training_model, train_loader, test_loader)
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.001, momentum=0.5)
    evaluator = TorchEvaluator(training_func, optimizer, training_step)  # type: ignore
    quantizer = DoReFaQuantizer(model, configure_list, evaluator)
    quantizer.compress()

    acc = evaluating_model(model, test_loader)
    print(f"inference: acc:{acc}")


if __name__ == '__main__':
    main()
