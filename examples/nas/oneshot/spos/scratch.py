# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from dataloader import get_imagenet_iter_dali
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.tensorboard import SummaryWriter

from network import ShuffleNetV2OneShot
from utils import CrossEntropyLabelSmooth, accuracy

logger = logging.getLogger("nni.spos.scratch")


def train(epoch, model, criterion, optimizer, loader, writer, args):
    model.train()
    meters = AverageMeterGroup()
    cur_lr = optimizer.param_groups[0]["lr"]

    for step, (x, y) in enumerate(loader):
        cur_step = len(loader) * epoch + step
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        metrics = accuracy(logits, y)
        metrics["loss"] = loss.item()
        meters.update(metrics)

        writer.add_scalar("lr", cur_lr, global_step=cur_step)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", metrics["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", metrics["acc5"], global_step=cur_step)

        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%d/%d] Step [%d/%d]  %s", epoch + 1,
                        args.epochs, step + 1, len(loader), meters)

    logger.info("Epoch %d training summary: %s", epoch + 1, meters)


def validate(epoch, model, criterion, loader, writer, args):
    model.eval()
    meters = AverageMeterGroup()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            logits = model(x)
            loss = criterion(logits, y)
            metrics = accuracy(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)

            if step % args.log_frequency == 0 or step + 1 == len(loader):
                logger.info("Epoch [%d/%d] Validation Step [%d/%d]  %s", epoch + 1,
                            args.epochs, step + 1, len(loader), meters)

    writer.add_scalar("loss/test", meters.loss.avg, global_step=epoch)
    writer.add_scalar("acc1/test", meters.acc1.avg, global_step=epoch)
    writer.add_scalar("acc5/test", meters.acc5.avg, global_step=epoch)

    logger.info("Epoch %d validation: top1 = %f, top5 = %f", epoch + 1, meters.acc1.avg, meters.acc5.avg)


def dump_checkpoint(model, epoch, checkpoint_dir):
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    dest_path = os.path.join(checkpoint_dir, "epoch_{}.pth.tar".format(epoch))
    logger.info("Saving model to %s", dest_path)
    torch.save(state_dict, dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Training From Scratch")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--tb-dir", type=str, default="runs")
    parser.add_argument("--architecture", type=str, default="architecture_final.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--lr-decay", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spos-preprocessing", default=False, action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = ShuffleNetV2OneShot(affine=True)
    model.cuda()
    apply_fixed_architecture(model, args.architecture)
    if torch.cuda.device_count() > 1:  # exclude last gpu, saving for data preprocessing on gpu
        model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count() - 1)))
    criterion = CrossEntropyLabelSmooth(1000, args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_decay == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda step: (1.0 - step / args.epochs)
                                                      if step <= args.epochs else 0,
                                                      last_epoch=-1)
    elif args.lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 1E-3)
    else:
        raise ValueError("'%s' not supported." % args.lr_decay)
    writer = SummaryWriter(log_dir=args.tb_dir)

    train_loader = get_imagenet_iter_dali("train", args.imagenet_dir, args.batch_size, args.workers,
                                          spos_preprocessing=args.spos_preprocessing)
    val_loader = get_imagenet_iter_dali("val", args.imagenet_dir, args.batch_size, args.workers,
                                        spos_preprocessing=args.spos_preprocessing)

    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, writer, args)
        validate(epoch, model, criterion, val_loader, writer, args)
        scheduler.step()
        dump_checkpoint(model, epoch, "scratch_checkpoints")

    writer.close()
