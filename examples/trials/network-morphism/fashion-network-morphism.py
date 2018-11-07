# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging
import os
import sys
import time

import utils
import nni
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# set the logger format
logger = logging.getLogger('fashion_mnist-network-morphism')
log_format = '%(asctime)s %(message)s'
logger.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0 


def get_args():
    parser = argparse.ArgumentParser("fashion_mnist")
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--epoches', type=int, default=200, help='epoch limit')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='epoch limit')
    parser.add_argument('--time_limit', type=int, default=0, help='gpu device id')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    args = parser.parse_args()
    return args

def build_graph(graph):
    net = graph
    return net


def prepare(graph,args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    # Data
    logger.info('Preparing data..')

    transform_train, transform_test = utils._data_transforms_cifar10(args)

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    # Model
    logger.info('Building model..')
    net = build_graph(graph)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()


    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    if args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    if args.optimizer == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate) 
         


# Training
def train(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    logger.info('Epoch: %d' % epoch)
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

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logger.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/fashion_best_model.pt')
        best_acc = acc
    return acc, best_acc

if __name__ == '__main__':
    try:
        args = get_args()
        # trial get next parameter from network morphism tuner
        RCV_CONFIG = nni.get_next_parameter()
        logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG,args)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(args.epoches):
            train(epoch)
            acc, best_acc = test(epoch)
            nni.report_intermediate_result(acc)

        # trial report best_acc to tuner
        nni.report_final_result(best_acc)
    except Exception as exception:
        logger.exception(exception)
        raise

