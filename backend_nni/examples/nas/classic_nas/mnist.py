"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
from collections import OrderedDict

import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture


logger = logging.getLogger('mnist_AutoML')


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        # two options of conv1
        self.conv1 = LayerChoice(OrderedDict([
            ("conv5x5", nn.Conv2d(1, 20, 5, 1)),
            ("conv3x3", nn.Conv2d(1, 20, 3, 1))
        ]), key='first_conv')
        # two options of mid_conv
        self.mid_conv = LayerChoice([
            nn.Conv2d(20, 20, 3, 1, padding=1),
            nn.Conv2d(20, 20, 5, 1, padding=2)
        ], key='mid_conv')
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        # skip connection over mid_conv
        self.input_switch = InputChoice(n_candidates=2,
                                        n_chosen=1,
                                        key='skip')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        old_x = x
        x = F.relu(self.mid_conv(x))
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = torch.add(x, skip_x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    hidden_size = args['hidden_size']

    model = Net(hidden_size=hidden_size).to(device)
    get_and_apply_next_architecture(model)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        if epoch < args['epochs']:
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')
        else:
            # report final result
            nni.report_final_result(test_acc)
            logger.debug('Final result is %g', test_acc)
            logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
