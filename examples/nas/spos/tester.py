import argparse
import logging
from itertools import cycle

import nni
import torch
import torch.nn as nn
from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture
from nni.nas.pytorch.utils import AverageMeterGroup
from torch.utils.data import DataLoader

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import get_imagenet, CrossEntropyLabelSmooth, accuracy

logger = logging.getLogger("nni")


def retrain_bn(model, criterion, max_iters, log_freq, loader_train, device):
    logger.info("Clear BN statistics...")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    logger.info("Train BN with training set (BN sanitize)...")
    model.train()
    meters = AverageMeterGroup()
    for step in range(max_iters):
        inputs, targets = next(loader_train)
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        metrics = accuracy(logits, targets)
        metrics["loss"] = loss.item()
        meters.update(metrics)
        if step % log_freq == 0 or step + 1 == max_iters:
            logger.info("Train Step [%d/%d] %s", step + 1, max_iters, meters)


def test_acc(model, criterion, max_iters, log_freq, loader_test, device):
    logger.info("Start testing...")
    model.eval()
    meters = AverageMeterGroup()
    with torch.no_grad():
        for step in range(max_iters):
            inputs, targets = next(loader_test)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            metrics = accuracy(logits, targets)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if step % log_freq == 0 or step + 1 == max_iters:
                logger.info("Valid Step [%d/%d] %s", step + 1, max_iters, meters)
    return meters.acc1.avg


def evaluate_acc(model, criterion, args, loader_train, loader_test, device):
    retrain_bn(model, criterion, args.train_iters, args.log_frequency, loader_train, device)
    acc = test_acc(model, criterion, args.test_iters, args.log_frequency, loader_test, device)
    assert isinstance(acc, float)
    nni.report_final_result(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Candidate Tester")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--checkpoint", type=str, default="./data/checkpoint-150000.pth.tar")
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--train-iters", type=int, default=128)
    parser.add_argument("--test-iters", type=int, default=40)
    parser.add_argument("--log-frequency", type=int, default=10)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    model = ShuffleNetV2OneShot()
    criterion = CrossEntropyLabelSmooth(1000, 0.1)
    get_and_apply_next_architecture(model)
    model.load_state_dict(load_and_parse_state_dict(filepath=args.checkpoint))
    model.to(device)

    dataset_train, dataset_valid = get_imagenet(args.imagenet_dir, spos_pre=args.spos_preprocessing)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_gpu)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=use_gpu)
    loader_train, loader_valid = cycle(loader_train), cycle(loader_valid)

    evaluate_acc(model, criterion, args, loader_train, loader_valid, device)
