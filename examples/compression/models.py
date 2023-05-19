# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models.resnet import resnet18

import nni

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mobilenet_v3():
    model = mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = torch.nn.Linear(1024, 10)
    return model.to(device)


def build_resnet18():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    return model.to(device)


def prepare_dataloader(batch_size: int = 128):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(
        datasets.CIFAR10(Path(__file__).parent / 'data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader


def prepare_optimizer(model: torch.nn.Module):
    optimize_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = nni.trace(Adam)(optimize_params, lr=0.001)
    return optimizer


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, training_step,
          lr_scheduler: _LRScheduler, max_steps: int, max_epochs: int):
    assert max_epochs is not None or max_steps is not None
    train_loader, test_loader = prepare_dataloader()
    max_steps = max_steps if max_steps else max_epochs * len(train_loader)
    max_epochs = max_steps // len(train_loader) + (0 if max_steps % len(train_loader) == 0 else 1)
    count_steps = 0

    model.train()
    for epoch in range(max_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = training_step((data, target), model)
            loss.backward()
            optimizer.step()
            count_steps += 1
            if count_steps >= max_steps:
                acc = evaluate(model, test_loader)
                print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')
                return
        acc = evaluate(model, test_loader)
        print(f'[Training Epoch {epoch} / Step {count_steps}] Final Acc: {acc}%')


def evaluate(model: torch.nn.Module, test_loader):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100 * correct / len(test_loader.dataset)


def training_step(batch, model: torch.nn.Module):
    output = model(batch[0])
    loss = F.cross_entropy(output, batch[1])
    return loss
