# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.data_utils import CIFAR10Policy, Cutout
from datasets.data_utils import SubsetDistributedSampler


def data_transforms_cifar(config, cutout=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if config.use_aa:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(), CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    if cutout:
        train_transform.transforms.append(Cutout(config.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def get_search_datasets(config):
    dataset = config.dataset.lower()
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = dset.CIFAR100
        n_classes = 100
    else:
        raise Exception("Not support dataset!")

    train_transform, valid_transform = data_transforms_cifar(config, cutout=False)
    train_data = dset_cls(root=config.data_dir, train=True, download=True, transform=train_transform)
    test_data = dset_cls(root=config.data_dir, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split_mid = int(np.floor(0.5 * num_train))

    if config.distributed:
        train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
        valid_sampler = SubsetDistributedSampler(train_data, indices[split_mid:num_train])
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split_mid])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split_mid:num_train])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=False, num_workers=config.workers)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=valid_sampler,
        pin_memory=False, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def get_augment_datasets(config):
    dataset = config.dataset.lower()
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
    elif dataset == 'cifar100':
        dset_cls = dset.CIFAR100
    else:
        raise Exception("Not support dataset!")

    train_transform, valid_transform = data_transforms_cifar(config, cutout=True)
    train_data = dset_cls(root=config.data_dir, train=True, download=True, transform=train_transform)
    test_data = dset_cls(root=config.data_dir, train=False, download=True, transform=valid_transform)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.eval_batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]
