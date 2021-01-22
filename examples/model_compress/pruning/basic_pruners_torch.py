# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
Examples for basic pruners
'''

import argparse

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms

from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG

import nni
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    FPGMPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationAPoZRankFilterPruner
)

str2pruner = {
    'level': LevelPruner,
    'l1_filter': L1FilterPruner,
    'l2_filter': L2FilterPruner,
    'slim': SlimPruner,
    'agp': AGPPruner,
    'fpgm': FPGMPruner,
    'apoz': ActivationAPoZRankFilterPruner
}

def get_pruner(model, pruner_name, device, optimizer=None, dependency_aware=False):
    def _get_dummy_input(args, device):
        if args.dataset == 'mnist':
            dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
        elif args.dataset in ['cifar10', 'imagenet']:
            dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
        return dummy_input

    pruner_cls = str2pruner[pruner_name]

    if pruner_name == 'level':
        op_types = ['default']
    elif pruner_name == 'slim':
        op_types = ['BatchNorm2d']
    else:
        op_types = ['Conv2d']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]

    kw_args = {}
    if dependency_aware:
        dummy_input = _get_dummy_input(args, device)
        print('Enable the dependency_aware mode')
        # note that, not all pruners support the dependency_aware mode
        kw_args['dependency_aware'] = True
        kw_args['dummy_input'] = dummy_input

    pruner = pruner_cls(model, config_list, optimizer, **kw_args)
    return pruner

def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data
    '''
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

    
def get_trained_model_optimizer(args, device, train_loader, test_loader, criterion):
    if args.model == 'LeNet':
        model = LeNet().to(device)
        if args.pretrained_model_dir is not None:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    elif args.model == 'vgg16':
        model = VGG(depth=16).to(device)
        if args.pretrained_model_dir is not None:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.pretrain_epochs*0.5), int(args.pretrain_epochs*0.75)], gamma=0.1)
    elif args.model == 'vgg19':
        model = VGG(depth=19).to(device)
        if args.pretrained_model_dir is not None:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(
                optimizer, milestones=[int(args.pretrain_epochs*0.5), int(args.pretrain_epochs*0.75)], gamma=0.1)
    else:
        raise ValueError("model not recognized")

    print('start pre-training...')
    if args.pretrained_model_dir is None:
        best_acc = 0
        best_epoch = 0
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(args, model, device, criterion, test_loader)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        print('Best acc:', best_acc)
        print('Best epoch:', best_epoch)

        torch.save(state_dict, os.path.join(args.experiment_data_dir, f'pretrain_{args.dataset}_{args.model}.pth'))
        print('Model trained saved to %s' % args.experiment_data_dir)

    return model, optimizer



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

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    model, optimizer = get_trained_model_optimizer(args, device, train_loader, test_loader, criterion).to(device)

    print('start pruning...')
    model_path = os.path.join(args.experiment_data_dir, 'pruned_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))
    mask_path = os.path.join(args.experiment_data_dir, 'mask_{}_{}_{}.pth'.format(
        args.model, args.dataset, args.pruner))

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    best_top1 = 0
    pruner = get_pruner(model, args.pruner, device, optimizer_finetune, args.dependency_aware)
    model = pruner.compress()

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(args.prune_epochs):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(args, model, device, train_loader, optimizer_finetune)
        top1 = test(args, model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path=model_path, mask_path=mask_path)

if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['LeNet', 'vgg16' ,'vgg19', 'resnet18'],
                        help='model to use')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    # pruner
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--pruner', type=str, default='l1_filter',
                        choices=['level', 'l1_filter', 'l2_filter', 'slim', 'agp',
                        'fpgm', 'apoz'],
                        help='pruner to use')

    # speed-up
    parser.add_argument('--speed-up', type=str2bool, default=False,
                        help='Whether to speed-up the pruned model')

    
    args = parser.parse_args()
    main(args)
