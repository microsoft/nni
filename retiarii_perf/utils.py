# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import pprint
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def accuracy(output, target, topk=(1, 5)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_logger(args):
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    def add_stdout_handler(logger):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if hasattr(args, "debug") and args.debug:
        # Debug log doesn't save
        add_stdout_handler(logger)
        logger.setLevel(logging.DEBUG)
    elif not hasattr(args, "local_rank") or args.local_rank == 0:
        # Process with local_rank > 0 will not produce any log
        add_stdout_handler(logger)
        logger.setLevel(logging.INFO)
        if not hasattr(args, "no_preserve_logs") or not args.no_preserve_logs:
            log_file = os.path.join(args.output_dir, "stdout.log")
            os.makedirs(args.output_dir, exist_ok=True)
            handler = logging.FileHandler(log_file, mode="w")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)
    logger.info("ARGPARSE: %s", json.dumps(vars(args)))

    logger.debug(pprint.pformat(vars(args)))
    return logger


def prepare_experiment(args):
    reset_seed(args.seed)

    args.tb_dir = os.path.join(args.output_dir, "tb")
    args.ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    # prepare_distributed(args)
    logger = prepare_logger(args)
    return logger


def prepare_distributed(args):
    if not hasattr(args, "distributed"):
        return
    logger = logging.getLogger("utils.distributed")
    if args.distributed:
        args.rank = int(os.environ.get("RANK", 0))
        # to be compatible with single worker mode
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        world_size = int(os.environ["WORLD_SIZE"])

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "54321")

        assert 0 <= args.local_rank < torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        args.world_size = world_size
        args.master_addr = master_addr
        args.master_port = master_port
        torch.distributed.init_process_group(backend="nccl", init_method="tcp://{}:{}".format(master_addr, master_port),
                                             world_size=world_size, rank=args.rank)
        args.is_worker_main = args.rank == 0
        args.is_worker_logging = args.local_rank == 0
    else:
        if "RANK" in os.environ:
            logger.warning("Rank is found in environment variables. Did you forget to set distributed?")
        args.is_worker_main = True
        args.is_worker_logging = True
        args.world_size = 1
        args.rank = args.local_rank = 0


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())

    def average_items(self):
        return {k: v.avg for k, v in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
