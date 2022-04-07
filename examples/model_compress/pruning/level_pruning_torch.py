# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported level pruning algorithm.
In this example, we show the end-to-end pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speedup is required.

'''
import argparse
import sys

import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from nni.compression.pytorch.utils import count_flops_params
from nni.compression.pytorch.pruning import LevelPruner

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from cifar10.vgg import VGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
g_epoch = 0

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

def trainer(model, optimizer, criterion):
    global g_epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx and batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                g_epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    g_epoch += 1

def evaluator(model):
    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100 * correct / len(test_loader.dataset)
    print('Accuracy: {}%\n'.format(acc))
    return acc

def optimizer_scheduler_generator(model, _lr=0.1, _momentum=0.9, _weight_decay=5e-4, total_epoch=160):
    optimizer = torch.optim.SGD(model.parameters(), lr=_lr, momentum=_momentum, weight_decay=_weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(total_epoch * 0.5), int(total_epoch * 0.75)], gamma=0.1)
    return optimizer, scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')
    parser.add_argument('--pretrain-epochs', type=int, default=20,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                        help='number of epochs to fine tune the model')
    args = parser.parse_args()

    print('\n' + '=' * 50 + ' START TO TRAIN THE MODEL ' + '=' * 50)
    model = VGG().to(device)
    optimizer, scheduler = optimizer_scheduler_generator(model, total_epoch=args.pretrain_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    pre_best_acc = 0.0
    best_state_dict = None

    for i in range(args.pretrain_epochs):
        trainer(model, optimizer, criterion)
        scheduler.step()
        acc = evaluator(model)
        if acc > pre_best_acc:
            pre_best_acc = acc
            best_state_dict = model.state_dict()
    print("Best accuracy: {}".format(pre_best_acc))
    model.load_state_dict(best_state_dict)
    pre_flops, pre_params, _ = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))

    # Start to prune and speedup
    print('\n' + '=' * 50 + ' START TO PRUNE THE BEST ACCURACY PRETRAINED MODEL ' + '=' * 50)
    config_list = [{
        'sparsity': 0.5,
        'op_types': ['default']
    }]
    pruner = LevelPruner(model, config_list)
    _, masks = pruner.compress()
    pruner.show_pruned_weights()

    # Fine-grained method does not need to speedup
    print('\n' + '=' * 50 + ' EVALUATE THE MODEL AFTER PRUNING ' + '=' * 50)
    evaluator(model)

    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    print('\n' + '=' * 50 + ' START TO FINE TUNE THE MODEL ' + '=' * 50)
    optimizer, scheduler = optimizer_scheduler_generator(model, _lr=0.01, total_epoch=args.fine_tune_epochs)

    best_acc = 0.0
    g_epoch = 0
    for i in range(args.fine_tune_epochs):
        trainer(model, optimizer, criterion)
        scheduler.step()
        best_acc = max(evaluator(model), best_acc)
    flops, params, results = count_flops_params(model, torch.randn([128, 3, 32, 32]).to(device))
    print(f'Pretrained model FLOPs {pre_flops/1e6:.2f} M, #Params: {pre_params/1e6:.2f}M, Accuracy: {pre_best_acc: .2f}%')
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}%')
