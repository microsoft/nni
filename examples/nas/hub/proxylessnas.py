# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import logging

import nni
import torch
from torch import nn

from nni.retiarii import strategy, fixed_arch
from nni.retiarii.evaluator.pytorch import Lightning, LightningModule, AccuracyWithLogits, Trainer
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.hub.pytorch import ProxylessNAS
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from timm.data import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT,
    create_dataset, create_loader
)
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.efficientnet_builder import efficientnet_init_weights
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn, ModelEmaV2
from torch.utils.data import DataLoader

_logger = logging.getLogger('nni')


@dataclass
class ImageNetTrainingHyperParameters:
    """Similar to the argument parser in timm:
    https://github.com/rwightman/pytorch-image-models/blob/f96da54eb1e03d7dfc32844deac34e231e73ea6f/train.py#L79

    Only necessary settings are kept here. Will add more when needed.
    """
    # Data parameters
    input_size: tuple[int, int, int] = (3, 224, 224)
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD
    interpolation: str = 'bicubic'
    crop_pct: float = DEFAULT_CROP_PCT
    workers: int = 4  # Number of workers for data loading

    # Model parameters
    batch_size: int = 128
    validation_batch_size: int | None = None
    init_weight_google: bool = False  # Weight initialization as per Tensorflow official implementations

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
    decay_milestones: list[int] = field(default_factory=lambda: [30, 60])  # decay epoch indices for multistep lr. must be increasing
    decay_epochs: int = 100  # epoch interval to decay LR
    warmup_epochs: int = 3  # epochs to warmup LR, if scheduler supports
    cooldown_epochs: int = 10  # epochs to cooldown LR at min_lr, after cyclic schedule ends
    patience_epochs: int = 10  # patience epochs for Plateau LR scheduler
    decay_rate: float = 0.1  # LR decay rate

    # Augmentation & regularization parameters
    scale: list[float] = field(default_factory=lambda: [0.08, 1.0])  # Random resize scale
    ratio: list[float] = field(default_factory=lambda: [3/4, 4/3])  # Random resize aspect ratio
    hflip: float = 0.5  # Horizontal flip training aug probability
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
    dist_bn: str | None = 'reduce'  # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce")

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

        if self.hparams.smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.hparams.smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.accuracy = AccuracyWithLogits()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(self.model, **optimizer_kwargs(cfg=self.hparams))

        # This is a hack. PyTorch-Lightning does not support schedulers with custom types.
        # Let's pretend there is not scheduler here. It will be manually called in `train_epoch_end`.
        self._lr_scheduler, self.num_epochs = create_scheduler(self.hparams, optimizer)
        _logger.info('Number of epochs: %d', self.num_epochs)
        return optimizer

    def set_model(self, model):
        super().set_model(model)

        if self.hparams.bn_momentum is not None or self.hparams.bn_eps is not None:
            # Reset BN momentum and epsilon if specified
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if self.hparams.bn_momentum is not None:
                        m.momentum = self.hparams.bn_momentum
                    if self.hparams.bn_eps is not None:
                        m.eps = self.hparams.bn_eps

        if self.hparams.init_weight_google:
            # Initialize weights per Tensorflow official implementations
            efficientnet_init_weights(self.model)

        if self.hparams.model_ema:
            # EMA will be wrapped by DDP in this case.
            self.model_ema = ModelEmaV2(self.model, decay=self.hparams.model_ema_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

        if self.hparams.model_ema:
            y_hat = self.model_ema(x)
        self.log('val_acc_ema', self.accuracy(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.log('test_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if self.hparams.model_ema:
            self.model_ema.update(self.model)
        return super().on_train_batch_end(outputs, batch, batch_idx, unused)

    def on_train_start(self):
        _logger.info('Start training at epoch: %d', self.current_epoch)
        if self.current_epoch > 0:
            self._lr_scheduler.step(self.current_epoch)

    def on_train_epoch_start(self):
        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self) -> None:
        # Update learning rate scheduler
        if self.hparams.model_ema:
            last_accuracy = self.trainer.callback_metrics['val_acc_ema']
        else:
            last_accuracy = self.trainer.callback_metrics['val_acc']
        if rank_zero_only.rank == 0:
            _logger.info('Schedule LR with metric: %s', last_accuracy)
        self._lr_scheduler.step(self.current_epoch + 1, last_accuracy)

        # Stop training when appropriate
        if self.current_epoch + 1 >= self.num_epochs:
            self.trainer.should_stop = True

        # Distribute Batch norm stats between nodes
        distributed = torch.distributed.is_initialized()

        if distributed and self.hparams.dist_bn in ('broadcast', 'reduce'):
            if rank_zero_only.rank == 0:
                _logger.info('Distributing BatchNorm running means and vars')
            distribute_bn(self.model, torch.distributed.get_world_size(), self.hparams.dist_bn == 'reduce')

        if self.hparams.model_ema:
            if distributed and self.hparams.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(self.model_ema, torch.distributed.get_world_size(), self.hparams.dist_bn == 'reduce')


def get_imagenet_dataloader(data_dir: str, hparams: ImageNetTrainingHyperParameters, train: bool) -> DataLoader:
    dataset = create_dataset('', data_dir, 'train' if train else 'validation')

    if train:
        return create_loader(
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
            interpolation=hparams.train_interpolation,
            mean=hparams.mean,
            std=hparams.std,
            num_workers=hparams.workers,
        )
    else:
        return create_loader(
            dataset,
            input_size=hparams.input_size,
            batch_size=hparams.validation_batch_size or hparams.batch_size,
            is_training=False,
            interpolation=hparams.interpolation,
            mean=hparams.mean,
            std=hparams.std,
            num_workers=hparams.workers,
            crop_pct=hparams.crop_pct,
        )


def train(arch: dict, log_dir: str, data_dir: str, batch_size: int | None = None, **kwargs):
    with fixed_arch(arch):
        model = ProxylessNAS(bn_momentum=0.01, bn_eps=0.001, dropout_rate=0.15)

    hparams = ImageNetTrainingHyperParameters()
    hparams.batch_size = 256
    hparams.lr = 2.64
    hparams.warmup_lr = 0.1
    hparams.warmup_epochs = 9
    hparams.epochs = 360
    hparams.sched = 'cosine'
    hparams.opt = 'rmsproptf'
    hparams.opt_eps = 1.
    hparams.weight_decay = 4e-5
    hparams.dist_bn = 'reduce'
    hparams.bn_momentum = 0.01
    hparams.bn_eps = 0.001

    # For debugging, you can use a smaller batch size
    if batch_size is not None:
        hparams.batch_size = batch_size

    train_loader = get_imagenet_dataloader(data_dir, hparams, True)
    valid_loader = get_imagenet_dataloader(data_dir, hparams, False)

    # evaluator = Lightning(
    #     TimmTrainingModule(hparams),
    #     Trainer(
    #         sync_batchnorm=hparams.sync_bn,
    #         logger=TensorBoardLogger(log_dir, name='train'),
    #     ),
    #     train_dataloaders=train_loader,
    #     val_dataloaders=valid_loader,
    # )

    # evaluator.fit(model)
    lightning_module = TimmTrainingModule(hparams)
    lightning_module.set_model(model)
    Trainer(gpus=1).validate(lightning_module, dataloaders=valid_loader)



def debug_train():
    train({
        's2_depth': 2,
        's2_i0': 'k5e3',
        's2_i1': 'k3e3',
        's3_depth': 4,
        's3_i0': 'k7e3',
        's3_i1': 'k3e3',
        's3_i2': 'k5e3',
        's3_i3': 'k5e3',
        's4_depth': 4,
        's4_i0': 'k7e6',
        's4_i1': 'k5e3',
        's4_i2': 'k5e3',
        's4_i3': 'k5e3',
        's5_depth': 4,
        's5_i0': 'k5e6',
        's5_i1': 'k5e3',
        's5_i2': 'k5e3',
        's5_i3': 'k5e3',
        's6_depth': 4,
        's6_i0': 'k7e6',
        's6_i1': 'k7e6',
        's6_i2': 'k7e3',
        's6_i3': 'k7e3',
        's7_depth': 1,
        's7_i0': 'k7e6'
    }, 'lightning_logs', 'data/imagenet', 256)


if __name__ == '__main__':
    debug_train()
