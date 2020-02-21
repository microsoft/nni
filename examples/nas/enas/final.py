# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter

logger = logging.getLogger("nni")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def train(config, train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_0.json")
    parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")

    args = parser.parse_args()

    dataset_train, dataset_valid = datasets.get_dataset("cifar10")
    if args.search_for == "macro":
        model = GeneralNetwork(out_filters=96, dropout_rate=0.5)
        args.batch_size = 100
        args.epochs = 310
    elif args.search_for == "micro":
        model = MicroNetwork(num_layers=15, out_channels=36, num_nodes=5, dropout_rate=0.2, use_aux_heads=False)
        args.epochs = 630
        args.batch_size = 144
    else:
        raise AssertionError

    apply_fixed_architecture(model, args.arc_checkpoint)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=2.0e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.001)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    best_top1 = 0.
    for epoch in range(args.epochs):
        # training
        train(args, train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(args, valid_loader, model, criterion, epoch, cur_step)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
