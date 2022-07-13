# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
import json

import nni
import numpy as np
import torch

from nni.retiarii import strategy, fixed_arch
from nni.retiarii.evaluator.pytorch import Lightning, ClassificationModule, LightningModule, AccuracyWithLogits, Trainer
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch import NasBench201

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from torch.utils.data import DataLoader



@dataclass
class ImageNetTrainingHyperParameters:
    """Similar to the argument parser in timm: https://github.com/rwightman/pytorch-image-models/blob/f96da54eb1e03d7dfc32844deac34e231e73ea6f/train.py#L79

    Only necessary settings are kept here. Will add more when needed.
    """
    # Data parameters
    input_size: tuple[int, int, int] = (3, 224, 224)
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None

    # Model parameters
    batch_size: int = 128
    validation_batch_size: int | None = None

    # Optimizer parameters
    opt: str = 'sgd'
    opt_eps: float | None = None
    momentum: float = 0.9
    weight_decay: float = 2e-5

    # Learning rate schedule parameters
    sched: str = 'cosine'
    lr: float = 0.05
    warmup_lr: float = 0.0001
    min_lr: float = 1e-6  # lower lr bound for cyclic schedulers that hit 0
    epochs: int = 300  # number of epochs sent to scheduler
    decay_milestones: list[int] = field(default=[30, 60])  # list of decay epoch indices for multistep lr. must be increasing
    decay_epochs: int = 100  # epoch interval to decay LR
    warmup_epochs: int = 3  # epochs to warmup LR, if scheduler supports
    cooldown_epochs: int = 10  # epochs to cooldown LR at min_lr, after cyclic schedule ends
    patience_epochs: int = 10  # patience epochs for Plateau LR scheduler
    decay_rate: float = 0.1  # LR decay rate

    # Augmentation & regularization parameters
    scale: list[float] = field(default=[0.08, 1.0])  # Random resize scale
    ratio: list[float] = field(default=[3/4, 4/3])  # Random resize aspect ratio
    hflip: list[float] = field(default=0.5)  # Horizontal flip training aug probability
    vflip: float = 0.  # Vertical flip training aug probability
    color_jitter: float = 0.4  # Color jitter factor
    aa: str = None  # Use AutoAugment policy. "v0" or "original".
    reprob: float = 0.  # Random erase prob
    remode: str = 'pixel'  # Random erase mode
    recount: int = 1  # Random erase count
    smoothing: float = 0.1  # Label smoothing
    train_interpolation: str = 'random'  # Training interpolation (random, bilinear, bicubic)
    drop: float = 0.0  # Dropout rate

    # Batch norm parameters
    bn_momentum: float | None = None
    bn_eps: float | None = None
    sync_bn: bool = False
    dist_bn: str = 'reduce'  # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")

    # Model Exponential Moving Average
    model_ema: bool = False
    model_ema_decay: float = 0.9998  # decay factor for model weights moving average


@nni.trace
class TimmTrainingModule(LightningModule):
    """Implementation of several features in https://github.com/rwightman/pytorch-image-models/blob/master/train.py
    with PyTorch-Lightning."""

    def __init__(self, hparams: ImageNetTrainingHyperParameters):
        super().__init__()
        self.save_hyperparameters(asdict(hparams))
        self.accuracy = AccuracyWithLogits()

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = create_optimizer_v2(self, **optimizer_kwargs(cfg=self.hparams))
        lr_scheduler, self.num_epochs = create_scheduler(self.hparams, optimizer)
        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler
        }

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('test_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch + 1 >= self.num_epochs:
            self.trainer.should_stop = True


def get_imagenet_dataloader(data_dir: str, hparams: ImageNetTrainingHyperParameters, train: bool) -> DataLoader:
    dataset = create_dataset('', data_dir, 'train' if train else 'validation', batch_size=hparams.batch_size)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset,
        input_size=hparams.input_size,
        batch_size=hparams.batch_size,
        is_training=True,
        re_prob=hparams.reprob,
        re_mode=hparams.remode,
        re_count=hparams.recount,
        scale=hparams.scale,
        ratio=hparams.ratio,
        hflip=hparams.hflip,
        vflip=hparams.vflip,
        color_jitter=hparams.color_jitter,
        auto_augment=hparams.aa,
        interpolation=train_interpolation,
        mean=hparams.mean,
        std=hparams.std,
        num_workers=hparams.workers,
        distributed=hparams.distributed,
        collate_fn=collate_fn,
        pin_memory=hparams.pin_mem,
        use_multi_epochs_loader=hparams.use_multi_epochs_loader,
        worker_seeding=hparams.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )