# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for simulated anealing pruning algorithm.
In this example, we show the end-to-end iterative pruning process: pre-training -> pruning -> fine-tuning.

'''
import sys
import argparse
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from nni.compression.pytorch.pruning import SimulatedAnnealingPruner

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from cifar10.vgg import VGG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

def trainer(model, optimizer, criterion, epoch):
    model.train()
    for data, target in tqdm(iterable=train_loader, desc='Epoch {}'.format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def finetuner(model):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for data, target in tqdm(iterable=train_loader, desc='Epoch PFs'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluator(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(iterable=test_loader, desc='Test'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print('Accuracy: {}%\n'.format(acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Iterative Example for model comporession')
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--pruning-algo', type=str, default='l1',
                        choices=['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz',
                                 'mean_activation', 'taylorfo', 'admm'],
                        help='algorithm to evaluate weights to prune')
    parser.add_argument('--cool-down-rate', type=float, default=0.9,
                        help='Cool down rate of the temperature.')

    args = parser.parse_args()

    model = VGG().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # pre-train the model
    for i in range(args.pretrain_epochs):
        trainer(model, optimizer, criterion, i)
        evaluator(model)

    config_list = [{'op_types': ['Conv2d'], 'total_sparsity': 0.8}]

    # evaluator in 'SimulatedAnnealingPruner' could not be None.
    pruner = SimulatedAnnealingPruner(model, config_list, pruning_algorithm=args.pruning_algo,
                                      evaluator=evaluator, cool_down_rate=args.cool_down_rate, finetuner=finetuner)
    pruner.compress()
    _, model, masks, _, _ = pruner.get_best_result()
    evaluator(model)
