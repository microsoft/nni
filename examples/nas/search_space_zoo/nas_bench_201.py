import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nni.nas.pytorch.utils import AverageMeterGroup
from nni.nas.pytorch.nas_bench_201.mutator import NASBench201Mutator
from nni.nas.pytorch.search_space_zoo import NASBench201Cell

from NASBench201Dataset import Nb201Dataset, get_dataloader
from .utils import nas_bench_201_accuracy


class ResNetBasicBlock(nn.Module):
    def __init__(self, configs, inplanes, planes, stride):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(configs, inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(configs, planes, planes, 3, 1, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(configs, inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        return inputs + basicblock


class NASBench201Network(nn.Module):
    def __init__(self, configs):
        super(NASBench201Network, self).__init__()
        self.channels = C = configs.stem_out_channels
        self.num_modules = N = configs.num_modules_per_stack
        self.num_labels = 10
        if configs.dataset.startswith("cifar100"):
            self.num_labels = 100
        if configs.dataset == "imagenet-16-120":
            self.num_labels = 120

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=configs.bn_momentum)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicBlock(configs, C_prev, C_curr, 2)
            else:
                cell = NASBench201Cell(configs, i, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=configs.bn_momentum),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


def train(model, loader, criterion, optimizer, scheduler, args, epoch):
    logger = logging.getLogger("nb201st.train.%d" % args.index)
    model.train()
    meters = AverageMeterGroup()

    logger.info("Current learning rate: %.6f", optimizer.param_groups[0]["lr"])
    for step, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        meters.update({"acc": nas_bench_201_accuracy(logits, targets), "loss": loss.item()})

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch, args.epochs, step + 1, len(loader), meters)
        scheduler.step()
    return meters.acc.avg, meters.loss.avg


def eval(model, split, loader, criterion, args, epoch):
    logger = logging.getLogger("nb201st.eval.%d" % args.index)
    model.eval()
    correct = loss = total = 0.
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            bs = targets.size(0)
            logits = model(inputs)
            loss += criterion(logits, targets).item() * bs
            correct += nas_bench_201_accuracy(logits, targets) * bs
            total += bs
    logger.info("%s Epoch [%d/%d] Loss = %.6f Acc = %.6f", split.capitalize(),
                epoch, args.epochs, loss / total, correct / total)
    return correct / total, loss / total


if __name__ == '__main__':
    args = parse_configs(Nb201Parser, "nb201st")
    logger = logging.getLogger("nb201st.main.%d" % args.index)

    train_loader, valid_loader, test_loader = get_dataloader(args)

    nasbench = Nb201Dataset(args.dataset, args.split, input_dtype=np.int8, acc_normalize=False)
    arch = nasbench[args.index]
    logger.info("Arch: %s", arch)
    model = NASBench201Network(args)
    mutator = NASBench201Mutator(model)
    mutator.reset(arch["matrix"])
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.resume_checkpoint, map_location="cpu"))
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optimizer.startswith("sgd"):
        optimizer = optim.SGD(model.parameters(), lr=args.initial_lr, momentum=args.momentum,
                              nesterov="nesterov" in args.optimizer, weight_decay=args.weight_decay)
    else:
        raise ValueError
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader),
                                                         eta_min=args.ending_lr)
    else:
        raise ValueError

    if args.keep_checkpoint != "none":
        os.makedirs(args.ckpt_dir, exist_ok=True)

    csv_records = []
    for epoch in range(1, args.epochs + 1):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, scheduler, args, epoch)
        val_acc, val_loss = eval(model, "val", valid_loader, criterion, args, epoch)
        csv_records.append({"epoch": epoch - 1, "accuracy": train_acc, "loss": train_loss,
                            "val_accuracy": val_acc, "val_loss": val_loss})
        if args.keep_checkpoint == "epoch":
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "epoch_%06d.pth.tar" % epoch))

    pd.DataFrame.from_records(csv_records).to_csv(os.path.join(args.output_dir, "training.csv"), index=False)
    eval(model, "test", test_loader, criterion, args, epoch)
    if args.keep_checkpoint == "end":
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "final.pth.tar"))
