import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cifar10.vgg import VGG
import nni
from nni.compression.torch import LevelPruner, SlimPruner, FPGMPruner, L1FilterPruner, \
    L2FilterPruner, AGPPruner, ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner

prune_config = {
    'level': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'pruner_class': LevelPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['default'],
        }]
    },
    'agp': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'pruner_class': AGPPruner,
        'config_list': [{
            'initial_sparsity': 0.,
            'final_sparsity': 0.8,
            'start_epoch': 0,
            'end_epoch': 10,
            'frequency': 1,
            'op_types': ['Conv2d']
        }]
    },
    'slim': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg19',
        'pruner_class': SlimPruner,
        'config_list': [{
            'sparsity': 0.7,
            'op_types': ['BatchNorm2d']
        }]
    },
    'fpgm': {
        'dataset_name': 'mnist',
        'model_name': 'naive',
        'pruner_class': FPGMPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']
        }]
    },
    'l1filter': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg16',
        'pruner_class': L1FilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
        }]
    },
    'mean_activation': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg16',
        'pruner_class': ActivationMeanRankFilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
        }]
    },
    'apoz': {
        'dataset_name': 'cifar10',
        'model_name': 'vgg16',
        'pruner_class': ActivationAPoZRankFilterPruner,
        'config_list': [{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': ['feature.0', 'feature.24', 'feature.27', 'feature.30', 'feature.34', 'feature.37']
        }]
    }
}


def get_data_loaders(dataset_name='mnist', batch_size=128):
    assert dataset_name in ['cifar10', 'mnist']

    if dataset_name == 'cifar10':
        ds_class = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.MNIST
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        ds_class = datasets.MNIST
        MEAN, STD = (0.1307,), (0.3081,)

    train_loader = DataLoader(
        ds_class(
            './data', train=True, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ds_class(
            './data', train=False, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        ),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_name='naive'):
    assert model_name in ['naive', 'vgg16', 'vgg19']

    if model_name == 'naive':
        return NaiveModel()
    elif model_name == 'vgg16':
        return VGG(16)
    else:
        return VGG(19)


def create_pruner(model, pruner_name, optimizer=None, dependency_aware=False, dummy_input=None):
    pruner_class = prune_config[pruner_name]['pruner_class']
    config_list = prune_config[pruner_name]['config_list']
    kw_args = {}
    if dependency_aware:
        print('Enable the dependency_aware mode')
        # note that, not all pruners support the dependency_aware mode
        kw_args['dependency_aware'] = True
        kw_args['dummy_input'] = dummy_input
    pruner = pruner_class(model, config_list, optimizer, **kw_args)
    return pruner

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('{:2.0f}%  Loss {}'.format(
                100 * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    model_name = prune_config[args.pruner_name]['model_name']
    dataset_name = prune_config[args.pruner_name]['dataset_name']
    train_loader, test_loader = get_data_loaders(dataset_name, args.batch_size)
    dummy_input, _ = next(iter(train_loader))
    dummy_input = dummy_input.to(device)
    model = create_model(model_name).cuda()
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print('loading checkpoint {} ...'.format(args.resume_from))
        model.load_state_dict(torch.load(args.resume_from))
        test(model, device, test_loader)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        if args.multi_gpu and torch.cuda.device_count():
            model = nn.DataParallel(model)

        print('start training')
        pretrain_model_path = os.path.join(
            args.checkpoints_dir, 'pretrain_{}_{}_{}.pth'.format(model_name, dataset_name, args.pruner_name))
        for epoch in range(args.pretrain_epochs):
            train(model, device, train_loader, optimizer)
            test(model, device, test_loader)
        torch.save(model.state_dict(), pretrain_model_path)

    print('start model pruning...')

    model_path = os.path.join(args.checkpoints_dir, 'pruned_{}_{}_{}.pth'.format(
        model_name, dataset_name, args.pruner_name))
    mask_path = os.path.join(args.checkpoints_dir, 'mask_{}_{}_{}.pth'.format(
        model_name, dataset_name, args.pruner_name))

    # pruner needs to be initialized from a model not wrapped by DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    best_top1 = 0

    pruner = create_pruner(model, args.pruner_name,
                           optimizer_finetune, args.dependency_aware, dummy_input)
    model = pruner.compress()

    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(args.prune_epochs):
        pruner.update_epoch(epoch)
        print('# Epoch {} #'.format(epoch))
        train(model, device, train_loader, optimizer_finetune)
        top1 = test(model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path=model_path, mask_path=mask_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruner_name", type=str,
                        default="level", help="pruner name")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int,
                        default=10, help="training epochs before model pruning")
    parser.add_argument("--prune_epochs", type=int, default=10,
                        help="training epochs for model pruning")
    parser.add_argument("--checkpoints_dir", type=str,
                        default="./checkpoints", help="checkpoints directory")
    parser.add_argument("--resume_from", type=str,
                        default=None, help="pretrained model weights")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multiple GPUs for training")
    parser.add_argument("--dependency_aware", action="store_true", default=False,
                        help="If enable the dependency_aware mode for the pruner")
    args = parser.parse_args()
    main(args)
