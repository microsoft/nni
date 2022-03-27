# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import torch
import torchvision

import numpy as np

from datasets import PFLDDatasets
from lib.builder import search_space
from lib.ops import PRIMITIVES
from lib.trainer import PFLDTrainer
from lib.utils import PFLDLoss
from nni.algorithms.nas.pytorch.fbnet import LookUpTable, NASConfig
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    """ The main function for supernet pre-training and subnet fine-tuning. """
    logging.basicConfig(
        format="[%(asctime)s] [p%(process)s] [%(pathname)s\
            :%(lineno)d] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # print the information of arguments
    for arg in vars(args):
        s = arg + ": " + str(getattr(args, arg))
        logging.info(s)

    # for 106 landmarks
    num_points = 106
    # list of device ids, and the number of workers for data loading
    device_ids = [int(id) for id in args.dev_id.split(",")]
    dev_num = len(device_ids)
    num_workers = 4 * dev_num

    # random seed
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    # import supernet for block-wise DNAS pre-training
    from lib.supernet import PFLDInference, AuxiliaryNet

    # the configuration for training control
    nas_config = NASConfig(
        perf_metric=args.perf_metric,
        lut_load=args.lut_load,
        model_dir=args.snapshot,
        nas_lr=args.theta_lr,
        mode=args.mode,
        alpha=args.alpha,
        beta=args.beta,
        search_space=search_space,
        start_epoch=args.start_epoch,
    )
    # look-up table with information of search space, flops per block, etc.
    lookup_table = LookUpTable(config=nas_config, primitives=PRIMITIVES)

    # create supernet
    pfld_backbone = PFLDInference(lookup_table, num_points)
    # the auxiliary-net of PFLD to predict the pose angle
    auxiliarynet = AuxiliaryNet()

    # main task loss
    criterion = PFLDLoss()

    # optimizer for weight train
    if args.opt == "adam":
        optimizer = torch.optim.AdamW(
            [
                {"params": pfld_backbone.parameters()},
                {"params": auxiliarynet.parameters()},
            ],
            lr=args.base_lr,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "rms":
        optimizer = torch.optim.RMSprop(
            [
                {"params": pfld_backbone.parameters()},
                {"params": auxiliarynet.parameters()},
            ],
            lr=args.base_lr,
            momentum=0.0,
            weight_decay=args.weight_decay,
        )

    # data argmentation and dataloader
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    # the landmark dataset with 106 points is default used
    train_dataset = PFLDDatasets(
        os.path.join(args.data_root, "train_data/list.txt"),
        transform,
        data_root=args.data_root,
        img_size=args.img_size,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_dataset = PFLDDatasets(
        os.path.join(args.data_root, "test_data/list.txt"),
        transform,
        data_root=args.data_root,
        img_size=args.img_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # create the trainer, then search/finetune
    trainer = PFLDTrainer(
        pfld_backbone,
        auxiliarynet,
        optimizer,
        criterion,
        device,
        device_ids,
        nas_config,
        lookup_table,
        dataloader,
        val_dataloader,
        n_epochs=args.end_epoch,
        logger=logging,
    )
    trainer.train()


def parse_args():
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    """ Parse the user arguments. """
    parser = argparse.ArgumentParser(description="FBNet for PFLD")
    parser.add_argument("--dev_id", dest="dev_id", default="0", type=str)
    parser.add_argument("--opt", default="rms", type=str)
    parser.add_argument("--base_lr", default=0.0001, type=int)
    parser.add_argument("--weight-decay", "--wd", default=1e-6, type=float)
    parser.add_argument("--img_size", default=112, type=int)
    parser.add_argument("--theta-lr", "--tlr", default=0.01, type=float)
    parser.add_argument(
        "--mode", default="mul", type=str, choices=["mul", "add"]
    )
    parser.add_argument("--alpha", default=0.25, type=float)
    parser.add_argument("--beta", default=0.6, type=float)
    parser.add_argument("--start_epoch", default=50, type=int)
    parser.add_argument("--end_epoch", default=300, type=int)
    parser.add_argument(
        "--snapshot", default="models", type=str, metavar="PATH"
    )
    parser.add_argument("--log_file", default="train.log", type=str)
    parser.add_argument(
        "--data_root", default="/dataset", type=str, metavar="PATH"
    )
    parser.add_argument("--train_batchsize", default=256, type=int)
    parser.add_argument("--val_batchsize", default=128, type=int)
    parser.add_argument(
        "--perf_metric", default="flops", type=str, choices=["flops", "latency"]
    )
    parser.add_argument(
        "--lut_load", type=str2bool, default=False
    )
    parser.add_argument(
        "--lut_load_format", default="json", type=str, choices=["json", "numpy"]
    )

    args = parser.parse_args()
    args.snapshot = os.path.join(args.snapshot, 'supernet')
    args.log_file = os.path.join(args.snapshot, "{}.log".format('supernet'))
    os.makedirs(args.snapshot, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
