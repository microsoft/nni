from __future__ import annotations
from typing import Callable, Any

import torch
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni
from nni.compression.pytorch import TorchEvaluator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from cifar10.vgg import VGG


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: torch.nn.Module = VGG().to(device)


def training_func(model: torch.nn.Module, optimizers: torch.optim.Optimizer,
                  criterion: Callable[[Any, Any], torch.Tensor],
                  lr_schedulers: _LRScheduler | None = None, max_steps: int | None = None,
                  max_epochs: int | None = None, *args, **kwargs):
    model.train()
    # prepare data
    cifar10_train_data = datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), download=True)
    train_dataloader = DataLoader(cifar10_train_data, batch_size=128, shuffle=True)

    total_epochs = max_epochs if max_epochs else 3
    total_steps = max_steps if max_steps else None
    current_steps = 0

    # training loop
    for _ in range(total_epochs):
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizers.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizers.step()
            current_steps += 1
            if total_steps and current_steps == total_steps:
                return
        lr_schedulers.step()


def evaluating_func(model: torch.nn.Module):
    model.eval()
    # prepare data
    cifar10_val_data = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]), download=True)
    val_dataloader = DataLoader(cifar10_val_data, batch_size=4, shuffle=False)
    # testing loop
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(labels.view_as(preds)).sum().item()
    return correct / len(cifar10_val_data)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
training_func(model, optimizer, criterion, lr_scheduler)
acc = evaluating_func(model)
print(f'The trained model accuracy: {acc}')

# create traced optimizer / lr_scheduler
optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = nni.trace(StepLR)(optimizer, step_size=1, gamma=0.5)
dummy_input = torch.rand(4, 3, 224, 224).to(device)

# TorchEvaluator initialization
evaluator = TorchEvaluator(training_func=training_func, optimizers=optimizer, criterion=criterion,
                           lr_schedulers=lr_scheduler, dummy_input=dummy_input, evaluating_func=evaluating_func)

# apply pruning
from nni.compression.pytorch.pruning import TaylorFOWeightPruner
from nni.compression.pytorch.speedup import ModelSpeedup

pruner = TaylorFOWeightPruner(model, config_list=[{'total_sparsity': 0.5, 'op_types': ['Conv2d']}], evaluator=evaluator, training_steps=100)
_, masks = pruner.compress()
acc = evaluating_func(model)
print(f'The masked model accuracy: {acc}')
pruner.show_pruned_weights()
pruner._unwrap_model()
ModelSpeedup(model, dummy_input=torch.rand([10, 3, 32, 32]).to(device), masks_file=masks).speedup_model()
acc = evaluating_func(model)
print(f'The speedup model accuracy: {acc}')

# finetune the speedup model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
training_func(model, optimizer, criterion, lr_scheduler)
acc = evaluating_func(model)
print(f'The speedup model after finetuning accuracy: {acc}')
