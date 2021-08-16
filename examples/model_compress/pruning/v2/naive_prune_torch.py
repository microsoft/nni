import argparse
import logging
from pathlib import Path

import torch
from torchvision import transforms, datasets

from nni.algorithms.compression.v2.pytorch import pruning
from nni.compression.pytorch import ModelSpeedup
from examples.model_compress.models.cifar10.vgg import VGG

logging.getLogger().setLevel(logging.DEBUG)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGG().to(device)

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=200, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

def trainer(model, optimizer, criterion, epoch=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def evaluator(model):
    model.eval()
    criterion = torch.nn.NLLLoss()
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

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
fintune_optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def main(args):
    if args.pre_train:
        for i in range(1):
            trainer(model, fintune_optimizer, criterion, epoch=i)

    config_list = [{
        'op_types': ['Conv2d'],
        'sparsity_per_layer': 0.8
    }]
    kwargs = {
        'model': model,
        'config_list': config_list,
    }
    if args.pruner == 'level':
        pruner = pruning.LevelPruner(**kwargs)
    else:
        kwargs['mode'] = args.mode
        if kwargs['mode'] == 'dependency_aware':
            kwargs['dummy_input'] = torch.rand(10, 3, 32, 32).to(device)
        if args.pruner == 'l1norm':
            pruner = pruning.L1NormPruner(**kwargs)
        elif args.pruner == 'l2norm':
            pruner = pruning.L2NormPruner(**kwargs)
        elif args.pruner == 'fpgm':
            pruner = pruning.FPGMPruner(**kwargs)
        else:
            kwargs['trainer'] = trainer
            kwargs['optimizer'] = optimizer
            kwargs['criterion'] = criterion
            if args.pruner == 'slim':
                kwargs['config_list'] = [{
                    'op_types': ['BatchNorm2d'],
                    'total_sparsity': 0.8,
                    'max_sparsity_per_layer': 0.9
                }]
                kwargs['training_epochs'] = 1
                pruner = pruning.SlimPruner(**kwargs)
            elif args.pruner == 'mean_activation':
                pruner = pruning.ActivationMeanRankPruner(**kwargs)
            elif args.pruner == 'apoz':
                pruner = pruning.ActivationAPoZRankPruner(**kwargs)
            elif args.pruner == 'taylorfo':
                pruner = pruning.TaylorFOWeightPruner(**kwargs)

    pruned_model, masks = pruner.compress()
    pruner.show_pruned_weights()

    if args.speed_up:
        tmp_masks = {}
        for name, mask in masks.items():
            tmp_masks[name] = {}
            tmp_masks[name]['weight'] = mask.get('weight_mask')
            if 'bias' in masks:
                tmp_masks[name]['bias'] = mask.get('bias_mask')
        torch.save(tmp_masks, Path('./temp_masks.pth'))
        pruner._unwrap_model()
        ModelSpeedup(model, torch.rand(10, 3, 32, 32).to(device), Path('./temp_masks.pth'))

    if args.finetune:
        for i in range(1):
            trainer(pruned_model, fintune_optimizer, criterion, epoch=i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example for model comporession')
    parser.add_argument('--pruner', type=str, default='l1norm',
                        choices=['level', 'l1norm', 'l2norm', 'slim',
                                 'fpgm', 'mean_activation', 'apoz', 'taylorfo'],
                        help='pruner to use')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'dependency_aware', 'global'])
    parser.add_argument('--pre-train', action='store_true', default=False,
                        help='Whether to pre-train the model')
    parser.add_argument('--speed-up', action='store_true', default=False,
                        help='Whether to speed-up the pruned model')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Whether to finetune the pruned model')
    args = parser.parse_args()

    main(args)
