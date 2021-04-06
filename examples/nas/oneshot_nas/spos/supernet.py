# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.nas.pytorch.callbacks import ModelCheckpoint
from nni.algorithms.nas.pytorch.spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer

from dataloader import get_imagenet_iter_dali
from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from utils import CrossEntropyLabelSmooth, accuracy

logger = logging.getLogger("nni.spos.supernet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = ShuffleNetV2OneShot()
    flops_func = model.get_candidate_flops
    if args.load_checkpoint:
        if not args.spos_preprocessing:
            logger.warning("You might want to use SPOS preprocessing if you are loading their checkpoints.")
        model.load_state_dict(load_and_parse_state_dict())
    model.cuda()
    if torch.cuda.device_count() > 1:  # exclude last gpu, saving for data preprocessing on gpu
        model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count() - 1)))
    mutator = SPOSSupernetTrainingMutator(model, flops_func=flops_func,
                                          flops_lb=290E6, flops_ub=360E6)
    criterion = CrossEntropyLabelSmooth(1000, args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.epochs else 0,
                                                  last_epoch=-1)
    train_loader = get_imagenet_iter_dali("train", args.imagenet_dir, args.batch_size, args.workers,
                                          spos_preprocessing=args.spos_preprocessing)
    valid_loader = get_imagenet_iter_dali("val", args.imagenet_dir, args.batch_size, args.workers,
                                          spos_preprocessing=args.spos_preprocessing)
    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer,
                                  args.epochs, train_loader, valid_loader,
                                  mutator=mutator, batch_size=args.batch_size,
                                  log_frequency=args.log_frequency, workers=args.workers,
                                  callbacks=[LRSchedulerCallback(scheduler),
                                             ModelCheckpoint("./checkpoints")])
    trainer.train()
