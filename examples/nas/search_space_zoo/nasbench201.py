import argparse
import json
import logging
import os
import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nni.nas.pytorch import enas
from nni.nas.pytorch.utils import AverageMeterGroup
from nni.nas.pytorch.nasbench201 import NASBench201Cell
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.benchmarks.nasbench201 import query_nb201_trial_stats
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from nni.nas.pytorch.darts import DartsTrainer
from utils import accuracy, reward_accuracy

import datasets

logger = logging.getLogger('nni')

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
                 bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=bn_affine, momentum=bn_momentum,
                           track_running_stats=bn_track_running_stats)
        )

    def forward(self, x):
        return self.op(x)


class ResNetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, bn_affine=True,
                 bn_momentum=0.1, bn_track_running_stats=True):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, bn_affine, bn_momentum, bn_track_running_stats)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, bn_affine, bn_momentum, bn_track_running_stats)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, bn_affine, bn_momentum, bn_track_running_stats)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        return inputs + basicblock


class NASBench201Network(nn.Module):
    def __init__(self, stem_out_channels, num_modules_per_stack, bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(NASBench201Network, self).__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = 10

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=self.bn_momentum)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicBlock(C_prev, C_curr, 2, self.bn_affine, self.bn_momentum, self.bn_track_running_stats)
            else:
                cell = NASBench201Cell(i, C_prev, C_curr, 1, self.bn_affine, self.bn_momentum, self.bn_track_running_stats)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=self.bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


def train(args, model, train_dataloader, valid_dataloader, criterion, optimizer, device):
    model = model.to(device)
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_frequency == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader), loss.item()))
        model.eval()
        correct = 0
        test_loss = 0.0
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(valid_dataloader.dataset)
        accuracy = 100. * correct / len(valid_dataloader.dataset)
        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                           len(valid_dataloader.dataset), accuracy))
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("nb201")
    parser.add_argument('--stem_out_channels', default=16, type=int)
    parser.add_argument('--unrolled', default=False, action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num_modules_per_stack', default=5, type=int)
    parser.add_argument('--log-frequency', default=10, type=int)
    parser.add_argument('--bn_momentum', default=0.1, type=int)
    parser.add_argument('--bn_affine', default=True, type=bool)
    parser.add_argument('--bn_track_running_stats', default=True, type=bool)
    parser.add_argument('--arch', default=None, help='json file which should meet requirements in NAS-Bench-201')
    parser.add_argument('--visualization', default=False, action='store_true')
    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    model = NASBench201Network(stem_out_channels=args.stem_out_channels,
                               num_modules_per_stack=args.num_modules_per_stack,
                               bn_affine=args.bn_affine,
                               bn_momentum=args.bn_momentum,
                               bn_track_running_stats=args.bn_track_running_stats)

    optim = torch.optim.SGD(model.parameters(), 0.025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)
    criterion = nn.CrossEntropyLoss()

    if args.arch is not None:
        logger.info('model retraining...')
        with open(args.arch, 'r') as f:
            arch = json.load(f)
        for trial in query_nb201_trial_stats(arch, 200, 'cifar100'):
            pprint.pprint(trial)
        apply_fixed_architecture(model, args.arch)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train(args, model, dataloader_train, dataloader_valid, criterion, optim,
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        exit(0)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                               reward_function=reward_accuracy,
                               optimizer=optim,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                               batch_size=args.batch_size,
                               num_epochs=args.epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequency=args.log_frequency)

    if args.visualization:
        trainer.enable_visualization()
    trainer.train()
