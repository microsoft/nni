# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""

"""
import logging

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from pruning.models.cifar10.resnet_cifar import resnet56
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch import ModelSpeedup
from nni.algorithms.compression.pytorch.pruning import (
    LevelPruner,
    SlimPruner,
    ADMMPruner,
    FPGMPruner,
    L1FilterPruner,
    L2FilterPruner,
    AGPPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer

str2pruner = {
    'level': LevelPruner,
    'l1filter': L1FilterPruner,
    'l2filter': L2FilterPruner,
    'slim': SlimPruner,
    'admm': ADMMPruner,
    'agp': AGPPruner,
    'fpgm': FPGMPruner,
    'mean_activation': ActivationMeanRankFilterPruner,
    'apoz': ActivationAPoZRankFilterPruner
}

def get_pruner(args, model, pruner_name, device, optimizer, trainer=None, dependency_aware=False):
    pruner_cls = str2pruner[pruner_name]
        
    config_list = [{
        'sparsity': args.sparsity,
        'op_types': ['Conv2d'],
        'op_names': [f'layer{i}.{j}.conv1' for i in range(1, 4) for j in range(9)]
    }]
    kw_args = {}
    if dependency_aware:
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
        print('Enable the dependency_aware mode')
        # note that, not all pruners support the dependency_aware mode
        kw_args['dependency_aware'] = True
        kw_args['dummy_input'] = dummy_input
    if pruner_name in ('slim', 'admm', 'agp'):
        if pruner_name == 'slim':
            config_list = [{
                'sparsity': args.sparsity,
                'op_types': ['BatchNorm2d'],
                'op_names': [f'layer{i}.{j}.bn1' for i in range(1, 4) for j in range(9)]
            }]
        elif pruner_name == 'agp':
            config_list = [{
            'initial_sparsity': 0.,
            'final_sparsity': args.sparsity,
            'start_epoch': 0,
            'end_epoch': 10,
            'frequency': 1,
            'op_names': [f'layer{i}.{j}.conv1']
        } for i in range(1, 4) for j in range(9)]
        else:
            kw_args['training_epochs'] = 2
            kw_args['num_iterations'] = 2
        pruner = pruner_cls(model, config_list, optimizer=optimizer, trainer=trainer, **kw_args)
    else:
        pruner = pruner_cls(model, config_list, optimizer, **kw_args)

    return pruner

def get_model_optimizer_scheduler(args, device, lr, model=None, pretrained_model_dir=None):
    if model is None:
        model = resnet56().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(args.pretrain_epochs * 0.5), int(args.pretrain_epochs * 0.75)], gamma=0.1)

    if pretrained_model_dir is not None:
        state_dict = torch.load(pretrained_model_dir)
        model.load_state_dict(state_dict)

    return model, optimizer, scheduler

def get_data(dataset, data_dir, batch_size, test_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }
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
        batch_size=test_batch_size, shuffle=False, **kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    return train_loader, test_loader, criterion

def get_model_time_cost(model, dummy_input):
    model.eval()
    n_times = 100
    time_list = []
    for _ in range(n_times):
        torch.cuda.synchronize()
        tic = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    return sum(time_list)


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()

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

    print('Test Loss: {:.6f}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)
    train_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)

    # Step1. Model Pretraining
    model, optimizer, scheduler = get_model_optimizer_scheduler(args, device, lr=args.pretrain_lr, pretrained_model_dir=args.pretrained_model_dir)
    flops, params, _ = count_flops_params(model, (1, 3, 32, 32), verbose=False)

    if args.pretrained_model_dir is None:
        args.pretrained_model_dir = os.path.join(args.experiment_data_dir, f'pretrained.pth')

        best_acc = 0
        for epoch in range(args.pretrain_epochs):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            scheduler.step()
            acc = test(args, model, device, criterion, test_loader)
            if acc > best_acc:
                best_acc = acc
                state_dict = model.state_dict()

        model.load_state_dict(state_dict)
        torch.save(state_dict, args.pretrained_model_dir)
        print(f'Model saved to {args.pretrained_model_dir}')
    else:
        best_acc = test(args, model, device, criterion, test_loader)

    dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    time_cost = get_model_time_cost(model, dummy_input)
    # 125.49 M, 0.85M, 93.29, 1.1012
    print(f'Pretrained model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}, Time Cost: {time_cost}')

    # Step2. Model Pruning
    def trainer(model, optimizer, criterion, epoch, callback=None):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, callback=callback)

    pruner = get_pruner(args, model, args.pruner, device, optimizer, trainer, args.dependency_aware)
    model = pruner.compress()
    pruner.get_pruned_weights()

    mask_path = os.path.join(args.experiment_data_dir, 'mask.pth')
    model_path = os.path.join(args.experiment_data_dir, 'pruned.pth')
    pruner.export_model(model_path=model_path, mask_path=mask_path)

    # Step3. Model Speedup
    m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
    m_speedup.speedup_model()

    flops, params, _ = count_flops_params(model, dummy_input, verbose=False)
    acc = test(args, model, device, criterion, test_loader)
    time_cost = get_model_time_cost(model, dummy_input)
    print(f'Pruned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {acc: .2f}, Time Cost: {time_cost}')

    # Step4. Model Finetuning
    model, optimizer, scheduler = get_model_optimizer_scheduler(args, device, model=model, lr=args.finetune_lr)

    best_acc = 0
    for epoch in range(args.finetune_epochs):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(args, model, device, criterion, test_loader)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    save_path = os.path.join(args.experiment_data_dir, f'finetuned.pth')
    torch.save(state_dict, save_path)

    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    time_cost = get_model_time_cost(model, dummy_input)
    # FLOPs 28.48 M, #Params: 0.18M, Accuracy:  89.03, Time Cost: 1.03
    print(f'Finetuned model FLOPs {flops/1e6:.2f} M, #Params: {params/1e6:.2f}M, Accuracy: {best_acc: .2f}, Time Cost: {time_cost}')
    print(f'Model saved to {save_path}')

    # Step5. Model Quantization via QAT
    config_list = [{
        'quant_types': ['weight'],
        'quant_bits': {
            'weight': 8,
        }, # you can just use `int` here because all `quan_types` share same bits length, see config for `ReLu6` below.
        'op_types':['Conv2d', 'Linear']
    }, {
        'quant_types': ['output'],
        'quant_bits': 8,
        'quant_start_step': 1000,
        'op_types':['ReLU']
    }]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    quantizer = QAT_Quantizer(model, config_list, optimizer)
    quantizer.compress()

    # Step6. Finetuning Quantized Model
    best_acc = 0
    for epoch in range(args.finetune_epochs):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc = test(args, model, device, criterion, test_loader)
        if acc > best_acc:
            best_acc = acc
            state_dict = model.state_dict()

    # Step7. Model Speedup
    # TODO.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--pretrained-model-dir', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--pretrain-epochs', type=int, default=160,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--pretrain-lr', type=float, default=0.1,
                        help='learning rate to pretrain the model')
        
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
    parser.add_argument('--pruner', type=str, default='l1filter',
                        choices=['level', 'l1filter', 'l2filter', 'slim', 'agp',
                                 'fpgm', 'mean_activation', 'apoz', 'admm'],
                        help='pruner to use')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='target overall target sparsity')
    parser.add_argument('--dependency-aware', action='store_true', default=False,
                        help='toggle dependency aware mode')

    # finetuning
    parser.add_argument('--finetune-epochs', type=int, default=80,
                        help='epochs to fine tune')
    parser.add_argument('--kd', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')
    parser.add_argument('--finetune-lr', type=float, default=0.01,
                        help='learning rate to finetune the model')

    # speedup
    parser.add_argument('--speed-up', action='store_true', default=False,
                        help='whether to speed-up the pruned model')

    parser.add_argument('--nni', action='store_true', default=False,
                        help="whether to tune the pruners using NNi tuners")

    args = parser.parse_args()
    main(args)


