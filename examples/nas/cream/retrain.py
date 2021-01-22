# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import os
import warnings
import datetime
import torch
import numpy as np
import torch.nn as nn

from torchscope import scope
from torch.utils.tensorboard import SummaryWriter

# import timm packages
from timm.optim import create_optimizer
from timm.models import resume_checkpoint
from timm.scheduler import create_scheduler
from timm.data import Dataset, create_loader
from timm.utils import ModelEma, update_summary
from timm.loss import LabelSmoothingCrossEntropy

# import apex as distributed package
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    HAS_APEX = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_APEX = False

# import models and training functions
from lib.core.test import validate
from lib.core.retrain import train_epoch
from lib.models.structures.childnet import gen_childnet
from lib.utils.util import parse_config_args, get_logger, get_model_flops_params
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    args, cfg = parse_config_args('nni.cream.childnet')

    # resolve logging
    output_dir = os.path.join(cfg.SAVE_PATH,
                              "{}-{}".format(datetime.date.today().strftime('%m%d'),
                                             cfg.MODEL))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.local_rank == 0:
        logger = get_logger(os.path.join(output_dir, 'retrain.log'))
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
        arch_list = [
            [0], [
                3, 3, 2, 3, 3], [
                3, 2, 3, 2, 3], [
                3, 2, 3, 2, 3], [
                    3, 3, 2, 2, 3, 3], [
                        3, 3, 2, 3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    elif cfg.NET.SELECTION == -1:
        arch_list = cfg.NET.INPUT_ARCH
        cfg.DATASET.IMAGE_SIZE = 224
    else:
        raise ValueError("Model Retrain Selection is not Supported!")

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

    # initialize training parameters
    eval_metric = cfg.EVAL_METRICS
    best_metric, best_epoch, saver = None, None, None

    # initialize distributed parameters
    distributed = cfg.NUM_GPU > 1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    if args.local_rank == 0:
        logger.info(
            'Training on Process {} with {} GPUs.'.format(
                args.local_rank, cfg.NUM_GPU))

    # fix random seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get parameters and FLOPs of model
    if args.local_rank == 0:
        macs, params = get_model_flops_params(model, input_size=(
            1, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
        logger.info(
            '[Model-{}] Flops: {} Params: {}'.format(cfg.NET.SELECTION, macs, params))

    # create optimizer
    model = model.cuda()
    optimizer = create_optimizer(cfg, model)

    # optionally resume from a checkpoint
    resume_state, resume_epoch = {}, None
    if cfg.AUTO_RESUME:
        resume_state, resume_epoch = resume_checkpoint(model, cfg.RESUME_PATH)
        optimizer.load_state_dict(resume_state['optimizer'])
        del resume_state

    model_ema = None
    if cfg.NET.EMA.USE:
        model_ema = ModelEma(
            model,
            decay=cfg.NET.EMA.DECAY,
            device='cpu' if cfg.NET.EMA.FORCE_CPU else '',
            resume=cfg.RESUME_PATH if cfg.AUTO_RESUME else None)

    if distributed:
        if cfg.BATCHNORM.SYNC_BN:
            try:
                if HAS_APEX:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                        model)
                if args.local_rank == 0:
                    logger.info(
                        'Converted model to use Synchronized BatchNorm.')
            except Exception as e:
                if args.local_rank == 0:
                    logger.error(
                        'Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1 with exception {}'.format(e))
        if HAS_APEX:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logger.info(
                    "Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            # can use device str in Torch >= 1.1
            model = DDP(model, device_ids=[args.local_rank])

    # imagenet train dataset
    train_dir = os.path.join(cfg.DATA_DIR, 'train')
    if not os.path.exists(train_dir) and args.local_rank == 0:
        logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)
    loader_train = create_loader(
        dataset_train,
        input_size=(3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size=cfg.DATASET.BATCH_SIZE,
        is_training=True,
        color_jitter=cfg.AUGMENTATION.COLOR_JITTER,
        auto_augment=cfg.AUGMENTATION.AA,
        num_aug_splits=0,
        crop_pct=DEFAULT_CROP_PCT,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=cfg.WORKERS,
        distributed=distributed,
        collate_fn=None,
        pin_memory=cfg.DATASET.PIN_MEM,
        interpolation='random',
        re_mode=cfg.AUGMENTATION.RE_MODE,
        re_prob=cfg.AUGMENTATION.RE_PROB
    )

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
        interpolation=cfg.DATASET.INTERPOLATION,
        crop_pct=DEFAULT_CROP_PCT,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=cfg.WORKERS,
        distributed=distributed,
        pin_memory=cfg.DATASET.PIN_MEM
    )

    # whether to use label smoothing
    if cfg.AUGMENTATION.SMOOTHING > 0.:
        train_loss_fn = LabelSmoothingCrossEntropy(
            smoothing=cfg.AUGMENTATION.SMOOTHING).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn

    # create learning rate scheduler
    lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)
    start_epoch = resume_epoch if resume_epoch is not None else 0
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)
    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(num_epochs))

    try:
        best_record, best_ep = 0, 0
        for epoch in range(start_epoch, num_epochs):
            if distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                cfg,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                model_ema=model_ema,
                logger=logger,
                writer=writer,
                local_rank=args.local_rank)

            eval_metrics = validate(
                epoch,
                model,
                loader_eval,
                validate_loss_fn,
                cfg,
                logger=logger,
                writer=writer,
                local_rank=args.local_rank)

            if model_ema is not None and not cfg.NET.EMA.FORCE_CPU:
                ema_eval_metrics = validate(
                    epoch,
                    model_ema.ema,
                    loader_eval,
                    validate_loss_fn,
                    cfg,
                    log_suffix='_EMA',
                    logger=logger,
                    writer=writer)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(epoch, train_metrics, eval_metrics, os.path.join(
                output_dir, 'summary.csv'), write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, cfg,
                    epoch=epoch, model_ema=model_ema, metric=save_metric)

            if best_record < eval_metrics[eval_metric]:
                best_record = eval_metrics[eval_metric]
                best_ep = epoch

            if args.local_rank == 0:
                logger.info(
                    '*** Best metric: {0} (epoch {1})'.format(best_record, best_ep))

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        logger.info(
            '*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


if __name__ == '__main__':
    main()
