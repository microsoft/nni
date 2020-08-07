'''Train CIFAR10 with PyTorch on AdaptDL.'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *
from utils import progress_bar

import nni

import adaptdl
import adaptdl.torch as et


_logger = logging.getLogger("cifar10_pytorch_automl")
IS_CHIEF = int(os.getenv("ADAPTDL_RANK", "0")) == 0
_logger.info("====> Is chief? " + str(IS_CHIEF))

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
best_acc = 0.0  # best test accuracy

def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    # Data
    print('==> Preparing data..')
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

    trainset = torchvision.datasets.CIFAR10(root=adaptdl.get_share_dir(), train=True, download=True, transform=transform_train)
    trainloader = et.ElasticDataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root=adaptdl.get_share_dir(), train=False, download=False, transform=transform_test)
    testloader = et.ElasticDataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    if args['model'] == 'vgg':
        net = VGG('VGG19')
    if args['model'] == 'resnet18':
        net = ResNet18()
    if args['model'] == 'googlenet':
        net = GoogLeNet()
    if args['model'] == 'densenet121':
        net = DenseNet121()
    if args['model'] == 'mobilenet':
        net = MobileNet()
    if args['model'] == 'dpn92':
        net = DPN92()
    if args['model'] == 'shufflenetg2':
        net = ShuffleNetG2()
    if args['model'] == 'senet18':
        net = SENet18()

    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    if args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    if args['optimizer'] == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'])

    net = et.ElasticDataParallel(model=net, optimizer=optimizer)


# Training
def train(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Update best_acc
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc


if __name__ == '__main__':
    # NOTE: auto size adjustment is only works on SGD for Adaptdl!!!
    # TODO: check tensorboard with `adaptdl._env.get_job_name()`
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.info(RCV_CONFIG)

        prepare(RCV_CONFIG)
        acc = 0.0
        best_acc = 0.0
        stats = et.Accumulator()
        for epoch in et.remaining_epochs_until(args.epochs):
            _logger.info("##############################")
            _logger.info(epoch)
            _logger.info("##############################")

            train(epoch)
            acc, best_acc = test(epoch)
            if IS_CHIEF:
                nni.report_intermediate_result(acc, stats)
        if IS_CHIEF:
            nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
