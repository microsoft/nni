# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Union, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni
from nni.compression.quantization import BNNQuantizer
from nni.compression.utils import TorchEvaluator
from nni.common.types import SCHEDULER


torch.manual_seed(0)
device = 'cuda'
cifar10_train = datasets.CIFAR10('./data.cifar10', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True)
cifar10_test = datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
test_loader = DataLoader(cifar10_test, batch_size=200, shuffle=False)


class VGG_Cifar10(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG_Cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),


            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, num_classes), # do not quantize output
            nn.BatchNorm1d(num_classes, affine=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def training_step(batch, model):
    data, target = batch[0].to(device), batch[1].to(device)
    output = model(data)
    return F.cross_entropy(output, target)


def test(model: nn.Module):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(cifar10_test)
    acc = 100 * correct / len(cifar10_test)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc


def train(model: torch.nn.Module, optimizer: Optimizer, training_step: Callable, scheduler: Union[SCHEDULER, None] = None,
          max_steps: Union[int, None] = None, max_epochs: Union[int, None] = 400):
    best_top1 = 0
    max_epochs = max_epochs or (40 if max_steps is None else 400)
    for epoch in range(max_epochs):
        print('# Epoch {} #'.format(epoch))
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = training_step(batch, model)
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, SCHEDULER):
                scheduler.step()
            if batch_idx % 100 == 0:
                print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))

        adjust_learning_rate(optimizer, epoch)
        top1 = test(model)
        if top1 > best_top1:
            best_top1 = top1
        print(f"epoch={epoch}\tcurrent_acc={top1}\tbest_acc={best_top1}")


def adjust_learning_rate(optimizer, epoch):
    update_list = [55, 100, 150, 200, 400, 600]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG_Cifar10(num_classes=10).to(device)
    configure_list = [{
        'op_names': ['features.3', 'features.7', 'features.10', 'features.14', 'classifier.0', 'classifier.3'],
        'target_names': ['weight'],
        'quant_dtype': None,
        'quant_scheme': "affine",
        'granularity': 'default',
    },
    {
        'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5'],
        'quant_dtype': None,
        'target_names': ['_output_'],
        'quant_scheme': "affine",
        'granularity': 'default',
    }]

    optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=1e-2)
    evaluator = TorchEvaluator(train, optimizer, training_step)  # type: ignore
    quantizer = BNNQuantizer(model, configure_list, evaluator)
    start = time.time()
    model, calibration_config = quantizer.compress(None, 400)
    end = time.time()
    print(f"time={end - start}")

    acc = test(model)
    print(f"inference: acc:{acc}")

    print(f"calibration_config={calibration_config}")


if __name__ == '__main__':
    main()
