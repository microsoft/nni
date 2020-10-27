# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import argparse
import time

import torch
import torch.nn as nn
from torchvision.models import resnet
from nni.compression.torch import AMCPruner
from data import get_split_dataset
from utils import AverageMeter, accuracy

sys.path.append('../models')

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')
    parser.add_argument('--model_type', default='mobilenet', type=str, choices=['mobilenet', 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50'],
        help='model to prune')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'], help='dataset to use (cifar/imagenet)')
    parser.add_argument('--batch_size', default=50, type=int, help='number of data batch size')
    parser.add_argument('--data_root', default='./data', type=str, help='dataset path')
    parser.add_argument('--flops_ratio', default=0.5, type=float, help='target flops ratio to preserve of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum sparsity')
    parser.add_argument('--rbound', default=1., type=float, help='maximum sparsity')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')

    parser.add_argument('--train_episode', default=800, type=int, help='number of training episode')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--suffix', default=None, type=str, help='suffix of auto-generated log directory')

    return parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if dataset == 'imagenet':
        n_class = 1000
    elif dataset == 'cifar10':
        n_class = 10
    else:
        raise ValueError('unsupported dataset')

    if model == 'mobilenet':
        from mobilenet import MobileNet
        net = MobileNet(n_class=n_class)
    elif model == 'mobilenetv2':
        from mobilenet_v2 import MobileNetV2
        net = MobileNetV2(n_class=n_class)
    elif model.startswith('resnet'):
        net = resnet.__dict__[model](pretrained=True)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, n_class)
    else:
        raise NotImplementedError
    if checkpoint_path:
        print('loading {}...'.format(checkpoint_path))
        sd = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if 'state_dict' in sd:  # a checkpoint but not a state_dict
            sd = sd['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        net.load_state_dict(sd)

    if torch.cuda.is_available() and n_gpu > 0:
        net = net.cuda()
        if n_gpu > 1:
            net = torch.nn.DataParallel(net, range(n_gpu))

    return net

def init_data(args):
    # split the train set into train + val
    # for CIFAR, split 5k for val
    # for ImageNet, split 3k for val
    val_size = 5000 if 'cifar' in args.dataset else 3000
    train_loader, val_loader, _ = get_split_dataset(
        args.dataset, args.batch_size,
        args.n_worker, val_size,
        data_root=args.data_root,
        shuffle=False
    )  # same sampling
    return train_loader, val_loader

def validate(val_loader, model, verbose=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    t1 = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = torch.autograd.Variable(input).to(device)
            target_var = torch.autograd.Variable(target).to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    t2 = time.time()
    if verbose:
        print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
              (losses.avg, top1.avg, top5.avg, t2 - t1))
    return top5.avg


if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() and args.n_gpu > 0 else torch.device('cpu')

    model = get_model_and_checkpoint(args.model_type, args.dataset, checkpoint_path=args.ckpt_path, n_gpu=args.n_gpu)
    _, val_loader = init_data(args)

    config_list = [{
        'op_types': ['Conv2d', 'Linear']
    }]
    pruner = AMCPruner(
        model, config_list, validate, val_loader, model_type=args.model_type, dataset=args.dataset,
        train_episode=args.train_episode, flops_ratio=args.flops_ratio, lbound=args.lbound,
        rbound=args.rbound, suffix=args.suffix)
    pruner.compress()
