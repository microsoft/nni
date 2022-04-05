# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
NNI example for supported basic pruning algorithms.
In this example, we show the end-to-end pruning process: pre-training -> pruning -> fine-tuning.
Note that pruners use masks to simulate the real pruning. In order to obtain a real compressed model, model speedup is required.
You can also try auto_pruners_torch.py to see the usage of some automatic pruning algorithms.

'''
import logging

import argparse
import os
import sys
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1] / 'models'))
from mnist.lenet import LeNet
from cifar10.vgg import VGG
from cifar10.resnet import ResNet18

from nni.compression.pytorch.utils.counter import count_flops_params

import nni
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    TaylorFOWeightFilterPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)

_logger = logging.getLogger('mnist_example')
_logger.setLevel(logging.INFO)

str2pruner = {
    'level': LevelPruner,
    'l1filter': L1FilterPruner,
    'l2filter': L2FilterPruner,
    'slim': SlimPruner,
    'agp': AGPPruner,
    'fpgm': FPGMPruner,
    'mean_activation': ActivationMeanRankFilterPruner,
    'apoz': ActivationAPoZRankFilterPruner,
    'taylorfo': TaylorFOWeightFilterPruner
}

def get_dummy_input(args, device):
    if args.dataset == 'mnist':
        dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
    elif args.dataset in ['cifar10', 'imagenet']:
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    return dummy_input


def get_data(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.NLLLoss()
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    return train_loader, test_loader, criterion

def get_model_optimizer_scheduler(args, device, train_loader, test_loader, criterion):
    if args.model == 'lenet':
        model = LeNet().to(device)
        if args.pretrained_model_dir is None:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    elif args.model == 'vgg16':
        model = VGG(depth=16).to(device)
        if args.pretrained_model_dir is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)
    elif args.model == 'vgg19':
        model = VGG(depth=19).to(device)
        if args.pretrained_model_dir is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)
    elif args.model == 'resnet18':
        model = ResNet18().to(device)
        if args.pretrained_model_dir is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)
    else:
        raise ValueError("model not recognized")

    if args.pretrained_model_dir is None:
        print('start pre-training...')
        best_acc = 0
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(args, model, device, criterion, test_loader)
            if acc > best_acc:
                best_acc = acc
                state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        acc = best_acc

        torch.save(state_dict, os.path.join(args.experiment_data_dir, f'pretrain_{args.dataset}_{args.model}.pth'))
        print('Model trained saved to %s' % args.experiment_data_dir)

    else:
        model.load_state_dict(torch.load(args.pretrained_model_dir))
        best_acc = test(args, model, device, criterion, test_loader)

    # setup new opotimizer for pruning
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)

    print('Pretrained model acc:', best_acc)
    return model, optimizer, scheduler

def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)

    model, optimizer, _ = get_model_optimizer_scheduler(args, device, train_loader, test_loader, criterion)

    dummy_input = get_dummy_input(args, device)
    flops, params, _ = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")

    print(f'start {args.pruner} pruning...')

    def trainer(model, optimizer, criterion, epoch):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch)

    pruner_cls = str2pruner[args.pruner]

    kw_args = {}
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Conv2d']
    }]

    if args.pruner == 'level':
        config_list = [{
            'sparsity': args.sparsity,
            'op_types': ['default']
        }]

    else:
        if args.global_sort:
            print('Enable the global_sort mode')
            # only taylor pruner supports global sort mode currently
            kw_args['global_sort'] = True
        if args.dependency_aware:
            dummy_input = get_dummy_input(args, device)
            print('Enable the dependency_aware mode')
            # note that, not all pruners support the dependency_aware mode
            kw_args['dependency_aware'] = True
            kw_args['dummy_input'] = dummy_input
        if args.pruner not in ('l1filter', 'l2filter', 'fpgm'):
            # set only work for training aware pruners
            kw_args['trainer'] = trainer
            kw_args['optimizer'] = optimizer
            kw_args['criterion'] = criterion

        if args.pruner in ('mean_activation', 'apoz', 'taylorfo'):
            kw_args['sparsifying_training_batches'] = 1

        if args.pruner == 'slim':
            kw_args['sparsifying_training_epochs'] = 1

        if args.pruner == 'agp':
            kw_args['pruning_algorithm'] = 'l1'
            kw_args['num_iterations'] = 2
            kw_args['epochs_per_iteration'] = 1

        # Reproduced result in paper 'PRUNING FILTERS FOR EFFICIENT CONVNETS',
        # Conv_1, Conv_8, Conv_9, Conv_10, Conv_11, Conv_12 are pruned with 50% sparsity, as 'VGG-16-pruned-A'
        # If you want to skip some layer, you can use 'exclude' like follow.
        if args.pruner == 'slim':
            config_list = [{
                'sparsity': args.sparsity,
                'op_types': ['BatchNorm2d'],
            }]
        elif args.model == 'resnet18':
            config_list = [{
                'sparsity': args.sparsity,
                'op_types': ['Conv2d']
            }, {
                'exclude': True,
                'op_names': ['layer1.0.conv1', 'layer1.0.conv2']
            }]
        else:
            config_list = [{
                'sparsity': args.sparsity,
                'op_types': ['Conv2d'],
                'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
            }]

    pruner = pruner_cls(model, config_list, **kw_args)

    # Pruner.compress() returns the masked model
    model = pruner.compress()
    pruner.get_pruned_weights()

    # export the pruned model masks for model speedup
    model_path = os.path.join(args.experiment_data_dir, 'pruned_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))
    mask_path = os.path.join(args.experiment_data_dir, 'mask_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))
    pruner.export_model(model_path=model_path, mask_path=mask_path)

    if args.test_only:
        test(args, model, device, criterion, test_loader)

    if args.speedup:
        # Unwrap all modules to normal state
        pruner._unwrap_model()
        m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
        m_speedup.speedup_model()

    print('start finetuning...')

    # Optimizer used in the pruner might be patched, so recommend to new an optimizer for fine-tuning stage.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)

    best_top1 = 0
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    for epoch in range(args.fine_tune_epochs):
        print('# Epoch {} #'.format(epoch))
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        top1 = test(args, model, device, criterion, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), save_path)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_top1: .2f}')

    if args.nni:
        nni.report_final_result(best_top1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['lenet', 'vgg16', 'vgg19', 'resnet18'],
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=160,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='input batch size for testing')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run on mulitple gpus')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='run test only')

    # pruner
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency-aware mode')
    parser.add_argument('--global-sort', action='store_true', default=False,
                        help='toggle global sort mode')
    parser.add_argument('--pruner', type=str, default='l1filter',
                        choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')

    # speedup
    parser.add_argument('--speedup', action='store_true', default=False,
                        help='Whether to speedup the pruned model')

    # fine-tuning
    parser.add_argument('--fine-tune-epochs', type=int, default=160,
                        help='epochs to fine tune')

    parser.add_argument('--nni', action='store_true', default=False,
                        help="whether to tune the pruners using NNi tuners")

    args = parser.parse_args()

    if args.nni:
        params = nni.get_next_parameter()
        print(params)
        args.sparsity = params['sparsity']
        args.pruner = params['pruner']
        args.model = params['model']

    main(args)