# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import os
import warnings
import datetime
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# import timm packages
from timm.utils import ModelEma
from timm.models import resume_checkpoint
from timm.data import Dataset, create_loader

# import apex as distributed package
try:
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP
    HAS_APEX = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_APEX = False

# import models and training functions
from lib.core.test import validate
from lib.models.structures.childnet import gen_childnet
from lib.utils.util import parse_config_args, get_logger, get_model_flops_params
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    args, cfg = parse_config_args('child net testing')

    # resolve logging
    output_dir = os.path.join(cfg.SAVE_PATH,
                              "{}-{}".format(datetime.date.today().strftime('%m%d'),
                                             cfg.MODEL))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.local_rank == 0:
        logger = get_logger(os.path.join(output_dir, 'test.log'))
        writer = SummaryWriter(os.path.join(output_dir, 'runs'))
    else:
        writer, logger = None, None

    # retrain model selection
    if cfg.NET.SELECTION == 481:
        arch_list = [
            [0], [
                3, 4, 3, 1], [
                3, 2, 3, 0], [
                3, 3, 3, 1], [
                    3, 3, 3, 3], [
                        3, 3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    elif cfg.NET.SELECTION == 43:
        arch_list = [[0], [3], [3, 1], [3, 1], [3, 3, 3], [3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 96
    elif cfg.NET.SELECTION == 14:
        arch_list = [[0], [3], [3, 3], [3, 3], [3], [3], [0]]
        cfg.DATASET.IMAGE_SIZE = 64
    elif cfg.NET.SELECTION == 112:
        arch_list = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 160
    elif cfg.NET.SELECTION == 287:
        arch_list = [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    elif cfg.NET.SELECTION == 604:
        arch_list = [[0], [3, 3, 2, 3, 3], [3, 2, 3, 2, 3], [3, 2, 3, 2, 3],
                     [3, 3, 2, 2, 3, 3], [3, 3, 2, 3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    else:
        raise ValueError("Model Test Selection is not Supported!")

    # define childnet architecture from arch_list
    stem = ['ds_r1_k3_s1_e1_c16_se0.25', 'cn_r1_k1_s1_c320_se0.25']
    choice_block_pool = ['ir_r1_k3_s2_e4_c24_se0.25',
                         'ir_r1_k5_s2_e4_c40_se0.25',
                         'ir_r1_k3_s2_e6_c80_se0.25',
                         'ir_r1_k3_s1_e6_c96_se0.25',
                         'ir_r1_k3_s2_e6_c192_se0.25']
    arch_def = [[stem[0]]] + [[choice_block_pool[idx]
                               for repeat_times in range(len(arch_list[idx + 1]))]
                              for idx in range(len(choice_block_pool))] + [[stem[1]]]

    # generate childnet
    model = gen_childnet(
        arch_list,
        arch_def,
        num_classes=cfg.DATASET.NUM_CLASSES,
        drop_rate=cfg.NET.DROPOUT_RATE,
        global_pool=cfg.NET.GP)

    if args.local_rank == 0:
        macs, params = get_model_flops_params(model, input_size=(
            1, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
        logger.info(
            '[Model-{}] Flops: {} Params: {}'.format(cfg.NET.SELECTION, macs, params))

    # initialize distributed parameters
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == 0:
        logger.info(
            "Training on Process {} with {} GPUs.".format(
                args.local_rank, cfg.NUM_GPU))

    # resume model from checkpoint
    assert cfg.AUTO_RESUME is True and os.path.exists(cfg.RESUME_PATH)
    _, __ = resume_checkpoint(model, cfg.RESUME_PATH)

    model = model.cuda()

    model_ema = None
    if cfg.NET.EMA.USE:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=cfg.NET.EMA.DECAY,
            device='cpu' if cfg.NET.EMA.FORCE_CPU else '',
            resume=cfg.RESUME_PATH)

    # imagenet validation dataset
    eval_dir = os.path.join(cfg.DATA_DIR, 'val')
    if not os.path.exists(eval_dir) and args.local_rank == 0:
        logger.error(
            'Validation folder does not exist at: {}'.format(eval_dir))
        exit(1)

    dataset_eval = Dataset(eval_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size=(3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size=cfg.DATASET.VAL_BATCH_MUL * cfg.DATASET.BATCH_SIZE,
        is_training=False,
        num_workers=cfg.WORKERS,
        distributed=True,
        pin_memory=cfg.DATASET.PIN_MEM,
        crop_pct=DEFAULT_CROP_PCT,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
    )

    # only test accuracy of model-EMA
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    validate(0, model_ema.ema, loader_eval, validate_loss_fn, cfg,
             log_suffix='_EMA', logger=logger,
             writer=writer, local_rank=args.local_rank)


if __name__ == '__main__':
    main()
