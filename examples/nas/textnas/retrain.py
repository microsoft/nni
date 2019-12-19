# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn

from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter

from model import Model
from utils import accuracy, get_data_loader


logger = logging.getLogger("nni.textnas")


def train(config, train_loader, model, optimizer, criterion, epoch):
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)

    model.train()
    for step, ((text, mask), y) in enumerate(train_loader):
        bs = text.size(0)
        optimizer.zero_grad()
        logits = model((text, mask))
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        acc = accuracy(logits, y)
        losses.update(loss.item(), bs)
        accs.update(acc, bs)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} {losses} {accs}".format(epoch + 1, config.epochs, step,
                                                                                      len(train_loader) - 1, losses=losses, accs=accs))

    logger.info("Train: [{:2d}/{}] Final Acc {:.2%}".format(epoch + 1, config.epochs, accs.avg))


def validate(config, valid_loader, model, criterion, epoch):
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")

    model.eval()
    with torch.no_grad():
        for step, ((text, mask), y) in enumerate(valid_loader):
            bs = text.size(0)
            logits = model((text, mask))
            loss = criterion(logits, y)
            acc = accuracy(logits, y)
            losses.update(loss.item(), bs)
            accs.update(acc, bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d} {losses} {accs}".format(epoch + 1, config.epochs, step,
                                                                                          len(valid_loader) - 1, losses=losses, accs=accs))

    logger.info("Valid: [{:2d}/{}] Final Acc {:.2%}".format(epoch + 1, config.epochs, accs.avg))
    return accs.avg


if __name__ == "__main__":
    parser = ArgumentParser("textnas")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=25, type=int)
    parser.add_argument("--arc-checkpoint", default="checkpoints/epoch_0.json", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embedding, train_loader, _, test_loader = get_data_loader(args.batch_size, device, infinite=False)
    model = Model(embedding)
    model.to(device)

    apply_fixed_architecture(model, args.arc_checkpoint)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, eps=1E-3, weight_decay=2E-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)

    best_top1 = 0.
    for epoch in range(args.epochs):
        train(args, train_loader, model, optimizer, criterion, epoch)
        top1 = validate(args, test_loader, model, criterion, epoch)
        best_top1 = max(best_top1, top1)
        lr_scheduler.step()

    logger.info("Best Acc = {:.4%}".format(best_top1))
