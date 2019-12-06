import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.compression.torch import SlimPruner
from models.cifar10.vgg import VGG


def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1


def train(model, device, train_loader, optimizer, sparse_bn=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # L1 regularization on BN layer
        if sparse_bn:
            updateBN(model)
        optimizer.step()
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


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
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

    model = VGG(depth=19)
    model.to(device)

    # Train the base VGG-19 model
    print('=' * 10 + 'Train the unpruned base model' + '=' * 10)
    epochs = 160
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    for epoch in range(epochs):
        if epoch in [epochs * 0.5, epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(model, device, train_loader, optimizer, True)
        test(model, device, test_loader)
    torch.save(model.state_dict(), 'vgg19_cifar10.pth')

    # Test base model accuracy
    print('=' * 10 + 'Test the original model' + '=' * 10)
    model.load_state_dict(torch.load('vgg19_cifar10.pth'))
    test(model, device, test_loader)
    # top1 = 93.60%

    # Pruning Configuration, in paper 'Learning efficient convolutional networks through network slimming',
    configure_list = [{
        'sparsity': 0.7,
        'op_types': ['BatchNorm2d'],
    }]

    # Prune model and test accuracy without fine tuning.
    print('=' * 10 + 'Test the pruned model before fine tune' + '=' * 10)
    pruner = SlimPruner(model, configure_list)
    model = pruner.compress()
    test(model, device, test_loader)
    # top1 = 93.55%

    # Fine tune the pruned model for 40 epochs and test accuracy
    print('=' * 10 + 'Fine tuning' + '=' * 10)
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    best_top1 = 0
    for epoch in range(40):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer_finetune)
        top1 = test(model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path='pruned_vgg19_cifar10.pth', mask_path='mask_vgg19_cifar10.pth')

    # Test the exported model
    print('=' * 10 + 'Test the export pruned model after fine tune' + '=' * 10)
    new_model = VGG(depth=19)
    new_model.to(device)
    new_model.load_state_dict(torch.load('pruned_vgg19_cifar10.pth'))
    test(new_model, device, test_loader)
    # top1 = 93.74%


if __name__ == '__main__':
    main()
