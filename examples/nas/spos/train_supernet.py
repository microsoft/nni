import argparse

import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import Callback, LRSchedulerCallback

from network import ShuffleNetV2OneShot
from src.sdk.pynni.nni.nas.pytorch.spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer
from utils import get_imagenet, CrossEntropyLabelSmooth, accuracy


class AdjustBNMomentum(Callback):
    def on_epoch_begin(self, epoch):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 1 / (epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--imagenet-dir", type=int, default="./data/imagenet")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4E-5)
    parser.add_argument("--label-smooth", type=float, default=0.1)

    args = parser.parse_args()
    dataset_train, dataset_valid = get_imagenet(args.imagenet_dir)
    model = ShuffleNetV2OneShot()
    model = nn.DataParallel(model)
    mutator = SPOSSupernetTrainingMutator(model, flops_func=model.module.get_candidate_flops,
                                          flops_lb=290E6, flops_ub=360E6)
    criterion = CrossEntropyLabelSmooth(1000, 0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0 - step / args.epochs)
                                                  if step <= args.total_iters else 0,
                                                  last_epoch=-1)
    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer,
                                  args.num_epochs, dataset_train, dataset_valid,
                                  mutator=mutator, batch_size=args.batch_size,
                                  callbacks=[LRSchedulerCallback(scheduler), AdjustBNMomentum()])
