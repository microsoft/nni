# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from typing import Callable, Union, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

from torchvision import datasets, transforms
from deepspeed import DeepSpeedEngine
from nni.compression.quantization import LsqQuantizer
from nni.compression.utils import DeepspeedTorchEvaluator
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


def training_step(batch, model) -> Tensor:
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss: torch.Tensor = F.nll_loss(logits, y)
    return loss


def training_model(model: DeepSpeedEngine, optimizer: Union[Optimizer, List[Optimizer]], \
                   training_step: _TRAINING_STEP, scheduler: Union[None, SCHEDULER, List[SCHEDULER]] = None,
                   max_steps: Union[int, None] = None, max_epochs: Union[int, None] = None):
    assert isinstance(model, DeepSpeedEngine)
    model.train()
    max_epochs = max_epochs or (40 if max_steps is None else 100)
    current_steps = 0
    best_acc = 0.0

    # training
    for epoch in range(max_epochs):
        print(f'Epoch {epoch} start!')
        model.train()
        for batch in train_dataloader:
            if isinstance(optimizer, Optimizer):
                optimizer.zero_grad()
            elif isinstance(optimizer, List) and all(isinstance(_, Optimizer) for _ in optimizer):
                for opt in optimizer:
                    opt.zero_grad()
            loss = training_step(batch, model)
            assert isinstance(loss, torch.Tensor)
            model.backward(loss)
            model.step()
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
    model = Mnist().to(device)
    configure_list = [{
        'target_names':['_input_', 'weight'],
        'op_names': ['conv2'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
    },{
        'op_names': ['relu1', 'relu2'],
        'target_names': ['_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
    },{
        'op_names': ['max_pool2'],
        'target_names': ['_output_'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
    },
    {
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['conv1'],
        'quant_dtype': 'int8',
        'quant_scheme': 'affine',
        'granularity': 'default',
        'fuse_names': [("conv1", "batchnorm1")]
    }]

    evaluator = DeepspeedTorchEvaluator(training_model, training_step, "./ds_config.json") #, lrs)
    quantizer = LsqQuantizer(model, configure_list, evaluator)
    model, calibration_config = quantizer.compress(None, 4)
    acc = evaluating_model(model)
    


if __name__ == '__main__':
    main()
