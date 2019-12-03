import argparse

import torch
import torch.nn as nn
from nni.nas.pytorch.callbacks import Callback, LRSchedulerCallback

from network import ShuffleNetV2OneShot, load_and_parse_state_dict
from nni.nas.pytorch.classic_nas import Class
from utils import get_imagenet, CrossEntropyLabelSmooth, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Candidate Tester")
    parser.add_argument("--imagenet-dir", type=str, default="./data/imagenet")
    parser.add_argument("--checkpoint", type=str, default=)
    parser.add_argument("--load-checkpoint", action="store_true", default=False)
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.5)
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
    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer,
                                  args.epochs, dataset_train, dataset_valid,
                                  mutator=mutator, batch_size=args.batch_size,
                                  log_frequency=args.log_frequency, workers=args.workers,
                                  callbacks=[LRSchedulerCallback(scheduler), AdjustBNMomentum()])
    trainer.train()
    # trainer.validate()
