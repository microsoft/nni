import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from ..common import graph_to_module_instance, ProxylessNASMixedOp
from ..searchspace import proxylessnas_gradient


def main(model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train_data = datasets.CIFAR10('./data/cifar10', download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    )

    criterion = nn.CrossEntropyLoss()
    model.cuda()

    mixed_ops = []
    for module in model.modules():
        if isinstance(module, ProxylessNASMixedOp):
            mixed_ops.append(module)

    model_parameters = [p for name, p in model.named_parameters()
                        if not name.endswith('._arch_parameters') and not name.endswith('._binary_gates')]
    arch_parameters = [p for name, p in model.named_parameters() if name.endswith('._arch_parameters')]
    weight_optimizer = torch.optim.SGD(model_parameters, 0.1, 0.9, 1e-4)
    arch_optimizer = optim.Adam(arch_parameters, 1e-4, betas=(0.5, 0.999), weight_decay=0)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=valid_sampler)

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
        val_X, val_y = val_X.cuda(), val_y.cuda()
        print('Running on step', step)

        # train architecture parameters
        for op in mixed_ops:
            op.resample()
        arch_optimizer.zero_grad()
        y_hat = model(val_X)
        loss = criterion(y_hat, val_y)
        loss.backward()
        for op in mixed_ops:
            op.finalize_grad()
        arch_optimizer.step()

        # train model parameters
        for op in mixed_ops:
            op.resample()
        arch_optimizer.zero_grad()
        weight_optimizer.zero_grad()
        y_hat = model(trn_X)
        loss = criterion(y_hat, trn_y)
        loss.backward()
        weight_optimizer.step()


def example():
    model_graph = proxylessnas_gradient()
    model = graph_to_module_instance(model_graph, ['examples.allstars.searchspace.proxylessnas',
                                                   'examples.allstars.common'])
    main(model)


if __name__ == '__main__':
    example()
