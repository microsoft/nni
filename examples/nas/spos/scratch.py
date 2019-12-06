import argparse
import logging

import torch
import torch.nn as nn
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.data.dataloader import DataLoader

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import get_imagenet, CrossEntropyLabelSmooth, accuracy

logger = logging.getLogger("nni")


def train(epoch, model, criterion, optimizer, loader, args):
    model.train()
    meters = AverageMeterGroup()
    for step, (x, y) in enumerate(loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        metrics = accuracy(logits, y)
        metrics["loss"] = loss.item()
        meters.update(metrics)
        if step % args.log_frequency == 0 or step + 1 == len(loader):
            logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                        args.epochs, step + 1, len(loader), meters)


def validate(epoch, model, criterion, loader, args):
    model.eval()
    meters = AverageMeterGroup()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            metrics = accuracy(logits, y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % args.log_frequency == 0 or step + 1 == len(loader):
                logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                            args.epochs, step + 1, len(loader), meters)
    logger.info("Epoch %d validation: %s", epoch + 1, meters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=1.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)

    args = parser.parse_args()
    dataset_train, dataset_valid = get_imagenet(args.imagenet_dir)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = ShuffleNetV2OneShot()
    model.cuda()
    apply_fixed_architecture(model, args.architecture)
    model = nn.DataParallel(model)
    criterion = CrossEntropyLabelSmooth(1000, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.epochs else 0,
                                                  last_epoch=-1)
    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, args)
        validate(epoch, model, criterion, valid_loader, args)
        scheduler.step()
