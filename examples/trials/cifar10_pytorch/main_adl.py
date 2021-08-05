# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Train CIFAR10 with PyTorch and AdaptDL. This example is based on:
https://github.com/petuum/adaptdl/blob/master/examples/pytorch-cifar/main.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import adaptdl
import adaptdl.torch as adl

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import nni


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=True, action='store_true', help='autoscale batchsize')
args = parser.parse_args()

# load the parameters from nni
RCV_CONFIG = nni.get_next_parameter()
args.lr = RCV_CONFIG["lr"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

if adaptdl.env.replica_rank() == 0:
    trainset = torchvision.datasets.CIFAR10(root=adaptdl.env.share_path(), train=True, download=True, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
    dist.barrier()  # We use a barrier here so that non-master replicas would wait for master to download the data
else:
    dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=adaptdl.env.share_path(), train=True, download=False, transform=transform_train)
    trainloader = adl.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

if args.autoscale_bsz:
    trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024), gradient_accumulation=True)

validset = torchvision.datasets.CIFAR10(root=adaptdl.env.share_path(), train=False, download=False, transform=transform_test)
validloader = adl.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = eval(args.model)()
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = MultiStepLR(optimizer, [30, 45], 0.1)

net = adl.AdaptiveDataParallel(net, optimizer, lr_scheduler)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    stats = adl.Accumulator()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item()

    trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data/")
    net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model/")
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        print("Train:", stats)

def valid(epoch):
    net.eval()
    stats = adl.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)

        if adaptdl.env.replica_rank() == 0:
            nni.report_intermediate_result(stats["accuracy"])

        print("Valid:", stats)
        return stats["accuracy"]


tensorboard_dir = os.path.join(
    os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/adaptdl/tensorboard"),
    os.getenv("NNI_TRIAL_JOB_ID", "cifar-adaptdl")
)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

with SummaryWriter(tensorboard_dir) as writer:
    acc = 0
    for epoch in adl.remaining_epochs_until(args.epochs):
        train(epoch)
        acc = valid(epoch)
        lr_scheduler.step()

    if adaptdl.env.replica_rank() == 0:
        nni.report_final_result(acc)
