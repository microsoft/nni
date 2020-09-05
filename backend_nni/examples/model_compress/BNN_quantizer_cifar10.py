# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.compression.torch import BNNQuantizer


class VGG_Cifar10(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG_Cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),


            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),


            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512, eps=1e-4, momentum=0.1),
            nn.Hardtanh(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Linear(1024, num_classes), # do not quantize output
            nn.BatchNorm1d(num_classes, affine=False)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        for name, param in model.named_parameters():
            if name.endswith('old_weight'):
                param = param.clamp(-1, 1)
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc

def adjust_learning_rate(optimizer, epoch):
    update_list = [55, 100, 150, 200, 400, 600]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=200, shuffle=False)

    model = VGG_Cifar10(num_classes=10)
    model.to(device)

    configure_list = [{
        'quant_types': ['weight'],
        'quant_bits': 1,
        'op_types': ['Conv2d', 'Linear'],
        'op_names': ['features.3', 'features.7', 'features.10', 'features.14', 'classifier.0', 'classifier.3']
    }, {
        'quant_types': ['output'],
        'quant_bits': 1,
        'op_types': ['Hardtanh'],
        'op_names': ['features.6', 'features.9', 'features.13', 'features.16', 'features.20', 'classifier.2', 'classifier.5']
    }]

    quantizer = BNNQuantizer(model, configure_list)
    model = quantizer.compress()

    print('=' * 10 + 'train' + '=' * 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    best_top1 = 0
    for epoch in range(400):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)
        adjust_learning_rate(optimizer, epoch)
        top1 = test(model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
    print(best_top1)


if __name__ == '__main__':
    main()
