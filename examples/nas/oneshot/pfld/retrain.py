# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time
import torch
import torchvision

import numpy as np

from datasets import PFLDDatasets
from lib.builder import search_space
from lib.ops import PRIMITIVES
from lib.utils import PFLDLoss, accuracy
from nni.algorithms.nas.pytorch.fbnet import (
    LookUpTable,
    NASConfig,
    supernet_sample,
)
from nni.nas.pytorch.utils import AverageMeter
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, auxiliarynet, valid_loader, device, logger):
    """Do validation."""
    model.eval()
    auxiliarynet.eval()

    losses, nme = list(), list()
    with torch.no_grad():
        for i, (img, land_gt, angle_gt) in enumerate(valid_loader):
            img = img.to(device, non_blocking=True)
            landmark_gt = land_gt.to(device, non_blocking=True)
            angle_gt = angle_gt.to(device, non_blocking=True)

            landmark, _ = model(img)

            # compute the l2 loss
            landmark = landmark.squeeze()
            l2_diff = torch.sum((landmark_gt - landmark) ** 2, axis=1)
            loss = torch.mean(l2_diff)
            losses.append(loss.cpu().detach().numpy())

            # compute the accuracy
            landmark = landmark.cpu().detach().numpy()
            landmark = landmark.reshape(landmark.shape[0], -1, 2)
            landmark_gt = landmark_gt.cpu().detach().numpy()
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2)
            _, nme_i = accuracy(landmark, landmark_gt)
            for item in nme_i:
                nme.append(item)

    logger.info("===> Evaluate:")
    logger.info(
        "Eval set: Average loss: {:.4f} nme: {:.4f}".format(
            np.mean(losses), np.mean(nme)
        )
    )
    return np.mean(losses), np.mean(nme)


def train_epoch(
    model,
    auxiliarynet,
    criterion,
    train_loader,
    device,
    epoch,
    optimizer,
    logger,
):
    """Train one epoch."""
    model.train()
    auxiliarynet.train()

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    losses = AverageMeter("losses")

    end = time.time()
    for i, (img, landmark_gt, angle_gt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.to(device, non_blocking=True)
        landmark_gt = landmark_gt.to(device, non_blocking=True)
        angle_gt = angle_gt.to(device, non_blocking=True)

        lands, feats = model(img)
        landmarks = lands.squeeze()
        angle = auxiliarynet(feats)

        # task loss
        weighted_loss, _ = criterion(
            landmark_gt, angle_gt, angle, landmarks
        )
        loss = weighted_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        losses.update(np.squeeze(loss.cpu().detach().numpy()), img.size(0))

        if i % 10 == 0:
            batch_log = (
                "Train [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {losses.val:.4f} ({losses.avg:.4f})".format(
                    epoch + 1,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    losses=losses,
                )
            )
            logger.info(batch_log)


def save_checkpoint(model, auxiliarynet, optimizer, filename, logger):
    """Save checkpoint of the whole model."""
    state = {
        "pfld_backbone": model.state_dict(),
        "auxiliarynet": auxiliarynet.state_dict(),
        "optim": optimizer.state_dict(),
    }
    torch.save(state, filename)
    logger.info("Save checkpoint to {0:}".format(filename))


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

    # import subnet for fine-tuning
    from lib.subnet import PFLDInference, AuxiliaryNet

    # the configuration for training control
    nas_config = NASConfig(
        model_dir=args.snapshot,
        search_space=search_space,
    )
    # look-up table with information of search space, flops per block, etc.
    lookup_table = LookUpTable(config=nas_config, primitives=PRIMITIVES)

    check = torch.load(args.supernet, map_location=torch.device("cpu"))
    sampled_arch = check["arch_sample"]
    logging.info(sampled_arch)
    # create subnet
    pfld_backbone = PFLDInference(lookup_table, sampled_arch, num_points)

    # pre-load the weights from pre-trained supernet
    state_dict = check["pfld_backbone"]
    supernet_sample(pfld_backbone, state_dict, sampled_arch, lookup_table)

    # the auxiliary-net of PFLD to predict the pose angle
    auxiliarynet = AuxiliaryNet()

    # DataParallel
    pfld_backbone = torch.nn.DataParallel(pfld_backbone, device_ids=device_ids)
    pfld_backbone.to(device)
    auxiliarynet = torch.nn.DataParallel(auxiliarynet, device_ids=device_ids)
    auxiliarynet.to(device)

    # main task loss
    criterion = PFLDLoss()

    # optimizer / scheduler for weight train
    optimizer = torch.optim.RMSprop(
        [
            {"params": pfld_backbone.parameters()},
            {"params": auxiliarynet.parameters()},
        ],
        lr=args.base_lr,
        momentum=0.0,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_epoch, last_epoch=-1
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

    # start finetune
    ckpt_path = args.snapshot
    val_nme = 1e6

    for epoch in range(0, args.end_epoch):
        logging.info("\n--------Train epoch: %d--------\n", epoch + 1)
        # update the weight parameters
        train_epoch(
            pfld_backbone,
            auxiliarynet,
            criterion,
            dataloader,
            device,
            epoch,
            optimizer,
            logging,
        )
        # adjust learning rate
        scheduler.step()

        # validate
        _, nme = validate(
            pfld_backbone, auxiliarynet, val_dataloader, device, logging
        )

        if epoch % 10 == 0:
            filename = os.path.join(ckpt_path, "checkpoint_%s.pth" % epoch)
            save_checkpoint(
                pfld_backbone, auxiliarynet, optimizer, filename, logging
            )

        if nme < val_nme:
            filename = os.path.join(ckpt_path, "checkpoint_best.pth")
            save_checkpoint(
                pfld_backbone, auxiliarynet, optimizer, filename, logging
            )
            val_nme = nme
        logging.info("Best nme: {:.4f}".format(val_nme))


def parse_args():
    """ Parse the user arguments. """
    parser = argparse.ArgumentParser(description="Finetuning for PFLD")
    parser.add_argument("--dev_id", dest="dev_id", default="0", type=str)
    parser.add_argument("--base_lr", default=0.0001, type=int)
    parser.add_argument("--weight-decay", "--wd", default=1e-6, type=float)
    parser.add_argument("--img_size", default=112, type=int)
    parser.add_argument("--supernet", default="", type=str, metavar="PATH")
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
    args = parser.parse_args()
    args.snapshot = os.path.join(args.snapshot, 'subnet')
    args.log_file = os.path.join(args.snapshot, "{}.log".format('subnet'))
    os.makedirs(args.snapshot, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
