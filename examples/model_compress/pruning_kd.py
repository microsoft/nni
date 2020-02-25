import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from nni.compression.torch import L1FilterPruner
from knowledge_distill.knowledge_distill import KnowledgeDistill
from models.cifar10.vgg import VGG


def train(model, device, train_loader, optimizer, kd=None):
    alpha = 1
    beta = 0.8
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        student_loss = F.cross_entropy(output, target)
        if kd is not None:
            kd_loss = kd.loss(data=data, student_out=output)
            loss = alpha * student_loss + beta * kd_loss
        else:
            loss = student_loss
        loss.backward()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = VGG(depth=16)
    model.to(device)

    # Train the base VGG-16 model
    print('=' * 10 + 'Train the unpruned base model' + '=' * 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 160, 0)
    for epoch in range(160):
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        lr_scheduler.step(epoch)
    torch.save(model.state_dict(), 'vgg16_cifar10.pth')

    # Test base model accuracy
    print('=' * 10 + 'Test on the original model' + '=' * 10)
    model.load_state_dict(torch.load('vgg16_cifar10.pth'))
    test(model, device, test_loader)
    # top1 = 93.51%

    # Pruning Configuration, all convolution layers are pruned out 80% filters according to the L1 norm
    configure_list = [{
        'sparsity': 0.8,
        'op_types': ['Conv2d'],
    }]

    # Prune model and test accuracy without fine tuning.
    print('=' * 10 + 'Test on the pruned model before fine tune' + '=' * 10)
    pruner = L1FilterPruner(model, configure_list)
    model = pruner.compress()
    test(model, device, test_loader)
    # top1 = 10.00%

    # Fine tune the pruned model for 40 epochs and test accuracy
    print('=' * 10 + 'Fine tuning' + '=' * 10)
    optimizer_finetune = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    best_top1 = 0
    kd_teacher_model = VGG(depth=16)
    kd_teacher_model.to(device)
    kd_teacher_model.load_state_dict(torch.load('vgg16_cifar10.pth'))
    kd = KnowledgeDistill(kd_teacher_model, kd_T=5)
    for epoch in range(40):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer_finetune, kd)
        top1 = test(model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path='pruned_vgg16_cifar10.pth', mask_path='mask_vgg16_cifar10.pth')

    # Test the exported model
    print('=' * 10 + 'Test on the pruned model after fine tune' + '=' * 10)
    new_model = VGG(depth=16)
    new_model.to(device)
    new_model.load_state_dict(torch.load('pruned_vgg16_cifar10.pth'))
    test(new_model, device, test_loader)
    # top1 = 85.43% with kd, top1 = 85.04% without kd,


if __name__ == '__main__':
    main()
