# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

import nni
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter

from dataloader import read_data_sst
from model import Model
from utils import accuracy


logger = logging.getLogger("nni.textnas")


def train(config, train_loader, model, optimizer, criterion, device, epoch):
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch + 1, cur_lr)

    model.train()
    for step, ((text, mask), y) in enumerate(train_loader):
        text, mask, y = text.to(device), mask.to(device), y.to(device)
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


def validate(config, valid_loader, model, criterion, device, epoch):
    losses = AverageMeter("loss")
    accs = AverageMeter("acc")

    model.eval()
    with torch.no_grad():
        for step, ((text, mask), y) in enumerate(valid_loader):
            text, mask, y = text.to(device), mask.to(device), y.to(device)            
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
    parser.add_argument("--log-frequency", default=50, type=int)
    parser.add_argument("--arc-checkpoint", default="final_arc.json", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    init_lr = 2E-3
    final_lr = 8E-4
    param = nni.get_next_parameter()
    if param:
        init_lr = param["init_lr"]
        final_lr = param["final_lr"]
        if final_lr > init_lr:
            raise ValueError

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, _, test_dataset, embedding = read_data_sst("data", train_with_valid=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    model = Model(embedding)
    model.to(device)

    apply_fixed_architecture(model, args.arc_checkpoint)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=1E-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=final_lr)

    best_top1 = top1 = 0.
    for epoch in range(args.epochs):
        train(args, train_loader, model, optimizer, criterion, device, epoch)
        top1 = validate(args, test_loader, model, criterion, device, epoch)
        nni.report_intermediate_result(top1)
        best_top1 = max(best_top1, top1)
        lr_scheduler.step()

    logger.info("Best Acc = {:.4%}".format(best_top1))
    nni.report_final_result(top1)
