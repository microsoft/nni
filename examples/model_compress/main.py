import os
import argparse
import logging
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import models.cifar10 as models
from torch.optim.lr_scheduler import MultiStepLR

from nni.compression.torch.utils import Config, configure_log_paths, AverageMeter, set_random_seed, accuracy, CSVLogger
import nni.compression.torch as nni_compression

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def update_from_args(args, config):
    for key, value in vars(args).items():
        config[key] = value
    return config

def create_pruner(model, optimizer_finetune, config):
    pruner = nni_compression.__dict__[config.pruner_name]
    prune_params = config['config_list']
    if 'Activation' in config.pruner_name or 'Gradient' in config.pruner_name:
        print(config.pruner_name)
        return pruner(model, prune_params, optimizer_finetune, statistics_batch_num=100)
    else:
        return pruner(model, prune_params, optimizer_finetune)

def main(args):
    print(args.config)
    config = Config(args.config)
    config = update_from_args(args, config)

    if config.model_name == 'resnet56':
        config['config_list'][0]['op_names'] = []
        if 'Slim' in config.pruner_name:
            op_type = 'bn1'
        else:
            op_type = 'conv1'

        for layer in range(1, 4):
            for block in range(9):
                config['config_list'][0]['op_names'].append(f'layer{layer}.{block}.{op_type}')
                
    elif config.model_name =='densenet40':
        config['config_list'][0]['op_names'] = []
        if 'Slim' in config.pruner_name:
            op_type = 'bn1'
        else:
            op_type = 'conv1'

        for layer in range(1, 4):
            config['config_list'][0]['op_names'].append(f'trans{layer}.conv1')
            for block in range(12):
                config['config_list'][0]['op_names'].append(f'dense{layer}.{block}.{op_type}')

    print(config)
    configure_log_paths(config)
    set_random_seed(config.random_seed, deterministic=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create data loader
    train_loader, test_loader = get_data_loaders(config)

    # Create model
    if config.model_name == 'naive':
        model = NaiveModel()
    else:
        model = models.__dict__[config.model_name]()
    model = model.to(device)

    if config.scratch:
        num_epochs = config.train_params['num_epochs']
        best_top1, best_top5 = 0, 0

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.5*num_epochs), int(0.75*num_epochs)], gamma=0.1)

        torch.save(model.state_dict(), f'{config.checkpoints_dir}/random_init.pth')

        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, config, epoch, num_epochs)
            top1, top5 = test(model, device, test_loader, config, epoch, num_epochs)
            lr_scheduler.step()

            if top1 > best_top1:
                best_top1 = top1
                best_top5 = top5
                torch.save(model.state_dict(), f'{config.checkpoints_dir}/pretrain.pth')

        print("## [Train from scratch] Top-1 {:.4%}\t Top-5 {:.4%}".format(best_top1, best_top5))

    elif config.pretrain:
        if os.path.isfile(config.pretrain):
            print('loading pretrained model {} ...'.format(config.pretrain))
            model.load_state_dict(torch.load(config.pretrain))
            # test(model, device, test_loader, config)
        else:
            print("no checkpoint found at '{}'".format(config.pretrain))
    
    if config.pruner_name == 'SlimPruner':
        print('=' * 10 + 'Train with sparsity' + '=' * 10)

        num_epochs = config['sparsity_epochs']
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        best_top1, best_top5 = 0, 0
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, config, epoch, num_epochs, sparse_bn=True)
            top1, top5 = test(model, device, test_loader, config, epoch, num_epochs)
            if top1 < best_top1:
                best_top1, best_top5 = top1, top5
                torch.save(model.state_dict(), f'{config.checkpoints_dir}/sparse_bn.pth')

        print("## [Train with sparsity] Top-1 {:.4%}\t Top-5 {:.4%}".format(best_top1, best_top5))

    # Create pruner
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    pruner = create_pruner(model, optimizer_finetune, config)
    model = pruner.compress()

    print('=' * 10 + 'Test on the pruned model before fine tune' + '=' * 10)
    test(model, device, test_loader, config)

    # Fine tune the pruned model and test accuracy
    print('=' * 10 + 'Fine tuning' + '=' * 10)
    num_epochs = config['retrain_epochs']

    best_top1, best_top5 = 0, 0
    csv_logger = CSVLogger(f'{config.log_dir}/results.csv')
    csv_logger.write(['epoch', 'top1', 'top5', 'lr'], reset=True)

    for epoch in range(num_epochs):
        pruner.update_epoch(epoch)
        train(model, device, train_loader, optimizer_finetune, config, epoch, num_epochs)
        top1, top5 = test(model, device, test_loader, config, epoch, num_epochs)

        cur_lr = get_lr(optimizer_finetune)
        csv_logger.write([epoch, top1, top5, cur_lr])

        if top1 > best_top1:
            best_top1, best_top5 = top1, top5
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path=f'{config.checkpoints_dir}/pruned.pth', mask_path=f'{config.checkpoints_dir}/mask.pth')

    print("## [Finetune Finish] Top-1 {:.4%}\t Top-5 {:.4%}".format(best_top1, best_top5))

def get_data_loaders(config):
    dataset_name = config.dataset_name
    assert dataset_name in ['cifar10', 'mnist']

    if dataset_name == 'cifar10':
        ds_class = datasets.CIFAR10 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        ds_class = datasets.MNIST
        transform_test = transform_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    batch_size = config.train_params['batch_size']
    train_loader = DataLoader(
        ds_class(
            config.dataset_path, train=True, download=True,
            transform=transform_train),batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ds_class(
            config.dataset_path, train=False, download=True,
            transform=transform_test),
            batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

def update_bn_params(model, sr):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(sr * torch.sign(m.weight.data))  # L1

def train(model, device, train_loader, optimizer, config, epoch, num_epochs, sparse_bn=False):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        cur_lr = get_lr(optimizer)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        if sparse_bn:
            update_bn_params(model, config.config_list[0]['sr'])

        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
        top5.update(acc5, data.size(0))

        if batch_idx % config.log_interval == 0:
            print(
                "Train: [{:3d}/{}]({:3d}/{:3d}) Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.2%}, {top5.avg:.2%}) "
                "Lr {cur_lr:.4f}".format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5, cur_lr=cur_lr))

def test(model, device, test_loader, config, epoch=0, num_epochs=1):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))

            if batch_idx % config.log_interval == 0:
                print(
                    "Test: [{:3d}/{}]({:3d}/{:3d})  Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.2%}, {top5.avg:.2%})".format(epoch + 1, num_epochs, batch_idx, len(test_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    print("* Test Final Prec@1 {:.2%}".format(top1.avg))

    return top1.avg, top5.avg

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file path")
    parser.add_argument("--random_seed", type=int, default=2333)
    parser.add_argument("--pruner_name", type=str, default="L1FilterPruner")
    parser.add_argument("--scratch", action='store_true', help="train the model from scratch")
    parser.add_argument("--pretrain", type=str, default=None, help="path to the pretrain model")
    parser.add_argument("--save_name", type=str, default='', help="name of the saved file")
    args = parser.parse_args()

    main(args)