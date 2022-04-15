# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
Examples for automatic pruners
'''

import argparse
import os
import json
import torch
import torchvision
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchvision import datasets, transforms

# from models.mnist.lenet import LeNet
from cifar_models.vgg import VGG
from cifar_models.mobilenetv2 import MobileNetV2
from cifar_models.resnet import ResNet18, ResNet50
import nni

from nni.algorithms.compression.pytorch.pruning import SimulatedAnnealingPruner, L1FilterPruner, L2FilterPruner, FPGMPruner
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency
import random
import time
import numpy as np


def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data
    '''
    kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.NLLLoss()
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion


def train(args, model, device, train_loader, criterion, optimizer, epoch, callback=None):
    _time_start = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    _time_end = time.time()
    _time_cost = _time_end - _time_start
    write_to_result('Training Time Cost:{}'.format( _time_cost))

def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


def get_trained_model_optimizer(args, device, train_loader, val_loader, criterion):
    if args.model == 'LeNet':
        model = LeNet().to(device)
        if args.load_pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-4)
    elif args.model == 'vgg16':
        if args.dataset == 'cifar10':
            model = VGG(depth=16).to(device)
        elif args.dataset == 'imagenet':
            model = torchvision.models.vgg16(pretrained=True).to(device)
        if args.load_pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    elif args.model == 'mobilenet_v2':
        if args.dataset == 'cifar10':
            model = MobileNetV2().to(device)
        elif args.dataset == 'imagenet':
            model = torchvision.models.mobilenet_v2(pretrained=True).to(device)
        if args.load_pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    elif args.model == 'resnet18':
        if args.dataset == 'cifar10':
            model = ResNet18().to(device)
        elif args.dataset == 'imagenet':
            model = torchvision.models.resnet18(pretrained=True).to(device)
        if args.load_pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    elif args.model == 'resnet50':
        if args.dataset == 'cifar10':
            model = ResNet50().to(device)
        elif args.dataset == 'imagenet':
            model = torchvision.models.resnet50(pretrained=True).to(device)
        if args.load_pretrained_model:
            model.load_state_dict(torch.load(args.pretrained_model_dir))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("model not recognized")

    acc = test(model, device, criterion, val_loader)
    print('Original Accuracy before the pruning', acc)
    return model, optimizer


def get_dummy_input(args, device):
    if args.dataset == 'mnist':
        dummy_input = torch.randn([args.test_batch_size, 1, 28, 28]).to(device)
    elif args.dataset == 'cifar10':
        dummy_input = torch.randn([args.test_batch_size, 3, 32, 32]).to(device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn([args.test_batch_size, 3, 224, 224]).to(device)
    return dummy_input

def get_input_size(dataset):
    if dataset == 'mnist':
        input_size = (1, 1, 28, 28)
    elif dataset in ['cifar10']:
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 224, 224)
    return input_size


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    model, optimizer = get_trained_model_optimizer(args, device, train_loader, val_loader, criterion)

    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, callback=callback)

    def evaluator(model):
        return test(model, device, criterion, val_loader)
    dummy_input = get_dummy_input(args, device)
    # used to save the performance of the original & pruned & finetuned models
    result = {'flops': {}, 'params': {}, 'performance':{}, 'time_mean':{}, 'time_std':{}}

    # flops, params = count_flops_params(model, get_input_size(args.dataset))
    # result['flops']['original'] = flops
    # result['params']['original'] = params
    evaluation_result = evaluator(model)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['performance']['original'] = evaluation_result

    # module types to prune, only "Conv2d" supported for channel pruning
    
    config_list = [{
        'sparsity': args.sparsity_per_iter,
        'op_types': ['Conv2d']
    }]

    sparsity_list = []
    for _iter in range(args.n_iter):
        # sparsity values if not enable the speedup
        sparsity_list.append((1-args.sparsity_per_iter)**(_iter+1))

    for _iter in range(args.n_iter):
        write_to_result("Pruning iteration {}".format(_iter))
        if not args.speed_up:
            config_list[0]['sparsity'] = sparsity_list[_iter]
        else:
            config_list[0]['sparsity'] = args.sparsity_per_iter
        print(config_list)
        pruner = SimulatedAnnealingPruner(
            model, config_list, evaluator=evaluator, cool_down_rate=args.cool_down_rate, 
            experiment_data_dir='./result', dummy_input=dummy_input, dependency_aware=args.constrained)
        model = pruner.compress()
        evaluation_result = evaluator(model)
        print('Evaluation result (masked model): %s' % evaluation_result)
        pruner.export_model('./model_ck.pth', './mask')

        if args.speed_up:
            pruner._unwrap_model()
            m_speedup = ModelSpeedup(model, dummy_input, './mask', device)
            m_speedup.speedup_model()

        if args.fine_tune:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            scheduler = None
            if args.lr_decay == 'multistep':
                scheduler = MultiStepLR(
                    optimizer, milestones=[int(args.fine_tune_epochs*0.25), int(args.fine_tune_epochs*0.5), int(args.fine_tune_epochs*0.75)], gamma=0.1)
            elif args.lr_decay == 'cos':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.fine_tune_epochs)
            best_acc = 0
            for epoch in range(args.fine_tune_epochs):
                acc = evaluator(model)
                print("acc at the begining", acc)
                train(args, model, device, train_loader, criterion, optimizer, epoch)
                if scheduler:
                    scheduler.step()
                acc = evaluator(model)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join('./result', 'model_fine_tuned.pth'))
        if not args.speed_up:
            # we need unwrap the model before get into the next iteration
            pruner._unwrap_model()
def write_to_result( line):
    with open('./Iterative_SA.log', 'a') as resf:
        resf.write(line+'\n')

if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to use, LeNet, vgg16, resnet18 or resnet50')
    parser.add_argument('--load-pretrained-model', type=str2bool, default=True,
                        help='whether to load pretrained model')
    parser.add_argument('--pretrained-model-dir', type=str, default='/root/checkpoints/cifar10/resnet50/model_trained.pth',
                        help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--fine-tune', type=str2bool, default=True,
                        help='whether to fine-tune the pruned model')
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='epochs to fine tune')
    parser.add_argument('--sparsity_per_iter', type=float, default=0.6,
                        help='target overall target sparsity')
    # param for SimulatedAnnealingPruner
    parser.add_argument('--cool-down-rate', type=float, default=0.97,
                        help='cool down rate')

    # speed-up
    parser.add_argument('--speed-up', type=str2bool, default=True,
                        help='Whether to speed-up the pruned model')

    # others
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str2bool, default=True,
                        help='For Saving the current Model')
    parser.add_argument('--constrained', type=str2bool, default=True, help='if enable the constraint-aware pruner')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate for the finetuning')
    parser.add_argument('--lr_decay', type=str, default='multistep', help='lr_decay type')
    parser.add_argument('--parallel', default=False, type=str2bool, help='If use multiple gpu to finetune the model')
    parser.add_argument('--aligned', default=1, type=int, help='The number of the pruned filter should be aligned with')
    parser.add_argument('--seed', default=2020, type=int, help='The random seed for torch and random module.')
    parser.add_argument('--n_iter', default=3, type=int, help='The number of pruning iteration')
    args = parser.parse_args()
    # random init the seed
    init_seed(args.seed)

    # hack the orignal L1FilterPruner
    dummy_input = get_dummy_input(args, 'cuda')
    _main_start = time.time()
    main(args)
    _main_end = time.time()
    _time_cost = _main_end - _main_start
    write_to_result( 'Total Time: {}\n'.format(_time_cost))
