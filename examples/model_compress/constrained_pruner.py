# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
Examples for automatic pruners
'''

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchvision import datasets, transforms, models

from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG
from nni.compression.torch import L1FilterPruner, Constrained_L1FilterPruner
from nni.compression.torch import L2FilterPruner, Constrained_L2FilterPruner
from nni.compression.torch import ActivationMeanRankFilterPruner, ConstrainedActivationMeanRankFilterPruner
from nni.compression.torch import ModelSpeedup
from nni.compression.torch.utils.counter import count_flops_params 

def cifar10_dataset(args):
    """
    return the train & test dataloader for the cifar10 dataset.
    """
    kwargs = {'num_workers': 10, 'pin_memory': True} if torch.cuda.is_available() else {
    }

   
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    dummy_input = torch.ones(1, 3, 32, 32)
    return train_loader, val_loader, dummy_input

def imagenet_dataset(args):
    kwargs = {'num_workers': 10, 'pin_memory': True} if torch.cuda.is_available() else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    dummy_input = torch.ones(1, 3, 224, 224)
    return train_loader, val_loader, dummy_input

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset to use, mnist, cifar10 or imagenet (default cifar10)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use, LeNet, vgg16, resnet18 or mobilenet_v2')
    parser.add_argument('--data-dir', type=str, default='/mnt/imagenet/raw_jpeg/2012/',
                        help='dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='overall target sparsity')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--finetune_epochs', type=int, default=15,
                        help='the number of finetune epochs after pruning')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate of model')
    parser.add_argument('--lr_decay', choices=['multistep', 'cos'], default='multistep', help='lr decay scheduler type')
    parser.add_argument('--type', choices=['l1', 'l2', 'activation'], default='l1', help='the pruning algo type')
    parser.add_argument('--para', action='store_true', help='if use multiple gpus')
    return parser.parse_args()


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_sum += loss.item()
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_sum/(batch_idx+1)))


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy

def get_data(args):
    if args.dataset == 'cifar10':
        return cifar10_dataset(args)
    elif args.dataset == 'imagenet':
        return imagenet_dataset(args)

if __name__ == '__main__':
    print("Benchmark the constraint-aware one shot pruner.")
    args = parse_args()
    torch.manual_seed(0)
    Model = getattr(models, args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, dummy_input = get_data(args)
    net1 = Model(pretrained=True).to(device)
    net2 = Model(pretrained=True).to(device)
    
    cfglist = [{'op_types':['Conv2d'], 'sparsity':args.sparsity}]
    if args.type == 'l1':
        pruner1 = L1FilterPruner(net1, cfglist)
        pruner2 = Constrained_L1FilterPruner(net2, cfglist, dummy_input.to(device))
    elif args.type == 'l2':
        pruner1 = L2FilterPruner(net1, cfglist)
        pruner2 = Constrained_L2FilterPruner(net2, cfglist, dummy_input.to(device))
    elif args.type == 'activation':
        pruner1 = ActivationMeanRankFilterPruner(net1, cfglist, statistics_batch_num=10)
        pruner2 = ConstrainedActivationMeanRankFilterPruner(net2, cfglist, dummy_input.to(device), statistics_batch_num=10)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            net1(data)
            net2(data)
            if batch_idx > 10:
                # enough data to calculate the activation
                break

    pruner1.compress()
    pruner2.compress()
    print('Model speedup finished')
    
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    scheduler1 = None
    scheduler2 = None
    if args.lr_decay == 'multistep':
        scheduler1 = MultiStepLR(
            optimizer1, milestones=[int(args.finetune_epochs*0.5), int(args.finetune_epochs*0.75)], gamma=0.1)
    elif args.lr_decay == 'cos':
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.finetune_epochs)
    criterion1 = torch.nn.CrossEntropyLoss()

    optimizer2 = torch.optim.SGD(net2.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    if args.lr_decay == 'multistep':
        scheduler2 = MultiStepLR(
            optimizer2, milestones=[int(args.finetune_epochs*0.5), int(args.finetune_epochs*0.75)], gamma=0.1)
    elif args.lr_decay == 'cos':
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.finetune_epochs)
    criterion2 = torch.nn.CrossEntropyLoss()

    acc1 = test(net1, device, criterion1, val_loader)
    acc2 = test(net2, device, criterion2, val_loader)
    print('After pruning: Acc of Original Pruner %f, Acc of Constrained Pruner %f' % (acc1, acc2))

    if args.para:
        net1 = nn.DataParallel(net1).to(device)
        net2 = nn.DataParallel(net2).to(device)
        # Scale the batch size, rebuild the data loader
        args.batch_size = args.batch_size * cuda.device_count()
        train_loader, val_loader, dummy_input = get_data(args)
    

    for epoch in range(args.finetune_epochs):
        train(args, net2, device, train_loader,
                criterion2, optimizer2, epoch)
        if scheduler2:
            scheduler2.step()
        acc2 = test(net2, device, criterion2, val_loader)
        print('Learning rate: ', scheduler2.get_last_lr())
        print('Finetune Epoch %d, acc of constrained pruner %f'%(epoch, acc2))

    for epoch in range(args.finetune_epochs):
        train(args, net1, device, train_loader,
                criterion1, optimizer1, epoch)
        if scheduler1:
            scheduler1.step()
        acc1 = test(net1, device, criterion1, val_loader)
        print('Learning rate: ', scheduler1.get_last_lr())
        print('Finetune Epoch %d, acc of original pruner %f'%(epoch, acc1))

    pruner1.export_model('./ori_%f.pth' % args.sparsity, './ori_mask_%f' % args.sparsity)
    pruner2.export_model('./cons_%f.pth' % args.sparsity, './cons_mask_%f' % args.sparsity)
    pruner1._unwrap_model()
    pruner2._unwrap_model()
    ms1 = ModelSpeedup(net1, dummy_input.to(device), './ori_mask_%f' % args.sparsity)
    ms2 = ModelSpeedup(net2, dummy_input.to(device), './cons_mask_%f' % args.sparsity)
    ms1.speedup_model()
    ms2.speedup_model()


    acc1 = test(net1, device, criterion1, val_loader)
    acc2 = test(net2, device, criterion2, val_loader)
    print('After finetuning: Acc of Original Pruner %f, Acc of Constrained Pruner %f' % (acc1, acc2))
    
    flops1, weights1 = count_flops_params(net1, dummy_input.size())
    flops2, weights2 = count_flops_params(net2, dummy_input.size())
    print('L1filter pruner flops:{} weight:{}'.format(flops1, weights1))
    print('Constrained L1filter pruner flops:{} weight:{}'.format(flops2, weights2))
    with open('result.txt', 'w') as f:
        f.write('L1filter pruner flops:{} weight:{} \n'.format(flops1, weights1))
        f.write('Constrained L1filter pruner flops:{} weight:{} \n'.format(flops2, weights2))
        f.write('After finetuning: Acc of Original Pruner %f, Acc of Constrained Pruner %f\n' % (acc1, acc2))