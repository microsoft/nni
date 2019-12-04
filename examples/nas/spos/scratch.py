import argparse
import logging

import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import Callback, LRSchedulerCallback

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from nni.nas.pytorch.callbacks import ModelCheckpoint
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=1.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)

    args = parser.parse_args()
    dataset_train, dataset_valid = get_imagenet(args.imagenet_dir, spos_pre=args.spos_preprocessing)
    model = ShuffleNetV2OneShot()
    if args.load_checkpoint:
        if not args.spos_preprocessing:
            print("You might want to use SPOS preprocessing if you are loading their checkpoints.")
        model.load_state_dict(load_and_parse_state_dict())
    model.cuda()
    apply_fixed_architecture(model, args.fixed_arc_path)
    model = nn.DataParallel(model)
    mutator = SPOSSupernetTrainingMutator(model, flops_func=model.module.get_candidate_flops,
                                          flops_lb=290E6, flops_ub=360E6)
    criterion = CrossEntropyLabelSmooth(1000, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.epochs else 0,
                                                  last_epoch=-1)
    for epoch in range(args.epochs):
        train(epoch, model, criterion, optimizer, train_loader, args)
        validate(epoch, model, criterion, test_loader, args)
        scheduler.step()
