# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
An exmaple for fine-tuning the pruend model with KD.
Run basic_pruners_torch.py first to get the pruend model.
'''

import argparse

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
from copy import deepcopy

from models.mnist.lenet import LeNet
from models.cifar10.vgg import VGG
from basic_pruners_torch import get_data

import nni
from nni.compression.pytorch import apply_compression_results, ModelSpeedup, get_dummy_input

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def get_model_optimizer_scheduler(args, device, test_loader, criterion):
    if args.model == 'LeNet':
        model = LeNet().to(device)
    elif args.model == 'vgg16':
        model = VGG(depth=16).to(device)
    elif args.model == 'vgg19':
        model = VGG(depth=19).to(device)
    else:
        raise ValueError("model not recognized")

    # In this example, we set the architecture of teacher and student to be the same. It is feasible to set a different teacher architecture.
    if args.teacher_model_dir is None:
        raise NotImplementedError('please load pretrained teacher model first')

    else:
        model.load_state_dict(torch.load(args.teacher_model_dir))
        best_acc = test(args, model, device, criterion, test_loader)

    model_t = deepcopy(model)
    model_s = deepcopy(model)

    if args.student_model_dir is not None:
        # load the pruned student model checkpoint
        model_s.load_state_dict(torch.load(args.student_model_dir))

    dummy_input = get_dummy_input(args, device)
    m_speedup = ModelSpeedup(model_s, dummy_input, args.mask_path, device)
    m_speedup.speedup_model()


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    module_list.append(model_t)

    # setup opotimizer for fine-tuning studeng model
    optimizer = torch.optim.SGD(model_s.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
                optimizer, milestones=[int(args.fine_tune_epochs*0.5), int(args.fine_tune_epochs*0.75)], gamma=0.1)
        
    print('Pretrained teacher model acc:', best_acc)
    return module_list, optimizer, scheduler


def train(args, models, device, train_loader, criterion, optimizer, epoch):
    # model.train()
    model_s = models[0].train()
    model_t = models[-1].eval()
    cri_cls = criterion
    cri_kd = DistillKL(args.kd_T)


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_s = model_s(data)
        output_t = model_t(data)

        loss_cls = cri_cls(output_s, target)
        loss_kd = cri_kd(output_s, output_t)
        loss = loss_cls + loss_kd
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(args, model, device, criterion, test_loader):
    model.eval()
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.experiment_data_dir, exist_ok=True)

    # prepare model and data
    train_loader, test_loader, criterion = get_data(args.dataset, args.data_dir, args.batch_size, args.test_batch_size)
    models, optimizer, scheduler = get_model_optimizer_scheduler(args, device, test_loader, criterion)

    best_top1 = 0
    if args.test_only:
        test(args, models[0], device, criterion, test_loader)

    print('start fine-tuning...')
    for epoch in range(args.fine_tune_epochs):
        print('# Epoch {} #'.format(epoch))
        train(args, models, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()

        # test student only
        top1 = test(args, models[0], device, criterion, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            torch.save(models[0].state_dict(), os.path.join(args.experiment_data_dir, 'model_trained.pth'))
            print('Model trained saved to %s' % args.experiment_data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='vgg16',
                        choices=['LeNet', 'vgg16' ,'vgg19', 'resnet18'],
                        help='model to use')
    parser.add_argument('--teacher-model-dir', type=str, default=None,
                        help='path to the pretrained teacher model checkpoint')
    parser.add_argument('--mask-path', type=str, default=None,
                        help='path to the pruend student model mask file')
    parser.add_argument('--student-model-dir', type=str, default=None,
                        help='path to the pruend student model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=200,
                        help='input batch size for testing')
    parser.add_argument('--fine-tune-epochs', type=int, default=160,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving output checkpoints')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='run test only')


    # knowledge distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')


    args = parser.parse_args()
    main(args)
