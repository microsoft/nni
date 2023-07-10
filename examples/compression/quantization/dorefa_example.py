# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from typing import Callable, Union, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

from torchvision import datasets, transforms

import nni
from nni.compression.quantization import DoReFaQuantizer
from nni.compression.utils import TorchEvaluator
from nni.common.types import SCHEDULER


torch.manual_seed(0)
device = 'cuda'
_TRAINING_STEP = Callable[..., Union[Tensor, Tuple[Tensor], Dict[str, Tensor]]]

datasets.MNIST(root='data/mnist', train=True, download=True)
datasets.MNIST(root='data/mnist', train=False, download=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='data/mnist', train=True, transform=transform)
train_dataloader = DataLoader(mnist_train, batch_size=64)
mnist_test = datasets.MNIST(root='data/mnist', train=False, transform=transform)
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


def training_step(batch, model) -> Tensor:
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss: torch.Tensor = F.nll_loss(logits, y)
    return loss


def training_model(model: torch.nn.Module, optimizer: Union[Optimizer, List[Optimizer]], \
                   training_step: _TRAINING_STEP, scheduler: Union[None, SCHEDULER, List[SCHEDULER]] = None,
                   max_steps: Union[int, None] = None, max_epochs: Union[int, None] = None):
    model.train()
    max_epochs = max_epochs or (10 if max_steps is None else 100)
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
            if isinstance(scheduler, SCHEDULER):
                scheduler.step()
            if isinstance(scheduler, List) and all(isinstance(_, SCHEDULER) for _ in scheduler):
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


def main():
    model = NaiveModel()
    model = model.to(device)
    configure_list = [{
    'op_names': ['conv1', 'conv2', 'fc1', 'fc2'],
    'target_names': ['_input_', 'weight'],
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
    optimizer = nni.trace(torch.optim.SGD)(model.parameters(), lr=0.001, momentum=0.5)
    evaluator = TorchEvaluator(training_model, optimizer, training_step)  # type: ignore
    quantizer = DoReFaQuantizer(model, configure_list, evaluator)
    model, calibration_config = quantizer.compress(None, 10)

    acc = evaluating_model(model)
    print(f"inference: acc:{acc}")

    print(f"calibration_config={calibration_config}")


if __name__ == '__main__':
    main()
