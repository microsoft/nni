# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import apex  # pylint: disable=import-error
from apex.parallel import DistributedDataParallel  # pylint: disable=import-error
from .mutator import RegularizedDartsMutator, RegularizedMutatorParallel, DartsDiscreteMutator  # pylint: disable=wrong-import-order
from nni.nas.pytorch.utils import AverageMeterGroup  # pylint: disable=wrong-import-order

from .utils import CyclicIterator, TorchTensorEncoder, accuracy, reduce_metrics

PHASE_SMALL = "small"
PHASE_LARGE = "large"


class InteractiveKLLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        # self.kl_loss = nn.KLDivLoss(reduction = 'batchmean')
        self.kl_loss = nn.KLDivLoss()

    def forward(self, student, teacher):
        return self.kl_loss(F.log_softmax(student / self.temperature, dim=1),
                            F.softmax(teacher / self.temperature, dim=1))


class CdartsTrainer(object):
    """
    CDARTS trainer.

    Parameters
    ----------
    model_small : nn.Module
        PyTorch model to be trained. This is the search network of CDARTS.
    model_large : nn.Module
        PyTorch model to be trained. This is the evaluation network of CDARTS.
    criterion : callable
        Receives logits and ground truth label, return a loss tensor, e.g., ``nn.CrossEntropyLoss()``.
    loaders : list of torch.utils.data.DataLoader
        List of train data and valid data loaders, for training weights and architecture weights respectively.
    samplers : list of torch.utils.data.Sampler
        List of train data and valid data samplers. This can be PyTorch standard samplers if not distributed.
        In distributed mode, sampler needs to have ``set_epoch`` method. Refer to data utils in CDARTS example for details.
    logger : logging.Logger
        The logger for logging. Will use nni logger by default (if logger is ``None``).
    regular_coeff : float
        The coefficient of regular loss.
    regular_ratio : float
        The ratio of regular loss.
    warmup_epochs : int
        The epochs to warmup the search network
    fix_head : bool
        ``True`` if fixing the paramters of auxiliary heads, else unfix the paramters of auxiliary heads.
    epochs : int
        Number of epochs planned for training.
    steps_per_epoch : int
        Steps of one epoch.
    loss_alpha : float
        The loss coefficient.
    loss_T : float
        The loss coefficient.
    distributed : bool
        ``True`` if using distributed training, else non-distributed training.
    log_frequency : int
        Step count per logging.
    grad_clip : float
        Gradient clipping for weights.
    interactive_type : string
        ``kl`` or ``smoothl1``.
    output_path : string
        Log storage path.
    w_lr : float
        Learning rate of the search network parameters.
    w_momentum : float
        Momentum of the search and the evaluation network.
    w_weight_decay : float
        The weight decay the search and the evaluation network parameters.
    alpha_lr : float
        Learning rate of the architecture parameters.
    alpha_weight_decay : float
        The weight decay the architecture parameters.
    nasnet_lr : float
        Learning rate of the evaluation network parameters.
    local_rank : int
        The number of thread.
    share_module : bool
        ``True`` if sharing the stem and auxiliary heads, else not sharing these modules.
    """
    def __init__(self, model_small, model_large, criterion, loaders, samplers, logger=None,
                 regular_coeff=5, regular_ratio=0.2, warmup_epochs=2, fix_head=True,
                 epochs=32, steps_per_epoch=None, loss_alpha=2, loss_T=2, distributed=True,
                 log_frequency=10, grad_clip=5.0, interactive_type='kl', output_path='./outputs',
                 w_lr=0.2, w_momentum=0.9, w_weight_decay=3e-4, alpha_lr=0.2, alpha_weight_decay=1e-4,
                 nasnet_lr=0.2, local_rank=0, share_module=True):
        if logger is None:
            logger = logging.getLogger(__name__)
        train_loader, valid_loader = loaders
        train_sampler, valid_sampler = samplers
        self.train_loader = CyclicIterator(train_loader, train_sampler, distributed)
        self.valid_loader = CyclicIterator(valid_loader, valid_sampler, distributed)

        self.regular_coeff = regular_coeff
        self.regular_ratio = regular_ratio
        self.warmup_epochs = warmup_epochs
        self.fix_head = fix_head
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        if self.steps_per_epoch is None:
            self.steps_per_epoch = min(len(self.train_loader), len(self.valid_loader))
        self.loss_alpha = loss_alpha
        self.grad_clip = grad_clip
        if interactive_type == "kl":
            self.interactive_loss = InteractiveKLLoss(loss_T)
        elif interactive_type == "smoothl1":
            self.interactive_loss = nn.SmoothL1Loss()
        self.loss_T = loss_T
        self.distributed = distributed
        self.log_frequency = log_frequency
        self.main_proc = not distributed or local_rank == 0

        self.logger = logger
        self.checkpoint_dir = output_path
        if self.main_proc:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if distributed:
            torch.distributed.barrier()

        self.model_small = model_small
        self.model_large = model_large
        if self.fix_head:
            for param in self.model_small.aux_head.parameters():
                param.requires_grad = False
            for param in self.model_large.aux_head.parameters():
                param.requires_grad = False

        self.mutator_small = RegularizedDartsMutator(self.model_small).cuda()
        self.mutator_large = DartsDiscreteMutator(self.model_large, self.mutator_small).cuda()
        self.criterion = criterion

        self.optimizer_small = torch.optim.SGD(self.model_small.parameters(), w_lr,
                                               momentum=w_momentum, weight_decay=w_weight_decay)
        self.optimizer_large = torch.optim.SGD(self.model_large.parameters(), nasnet_lr,
                                               momentum=w_momentum, weight_decay=w_weight_decay)
        self.optimizer_alpha = torch.optim.Adam(self.mutator_small.parameters(), alpha_lr,
                                                betas=(0.5, 0.999), weight_decay=alpha_weight_decay)

        if distributed:
            apex.parallel.convert_syncbn_model(self.model_small)
            apex.parallel.convert_syncbn_model(self.model_large)
            self.model_small = DistributedDataParallel(self.model_small, delay_allreduce=True)
            self.model_large = DistributedDataParallel(self.model_large, delay_allreduce=True)
            self.mutator_small = RegularizedMutatorParallel(self.mutator_small, delay_allreduce=True)
            if share_module:
                self.model_small.callback_queued = True
                self.model_large.callback_queued = True
            # mutator large never gets optimized, so do not need parallelized

    def _warmup(self, phase, epoch):
        assert phase in [PHASE_SMALL, PHASE_LARGE]
        if phase == PHASE_SMALL:
            model, optimizer = self.model_small, self.optimizer_small
        elif phase == PHASE_LARGE:
            model, optimizer = self.model_large, self.optimizer_large
        model.train()
        meters = AverageMeterGroup()
        for step in range(self.steps_per_epoch):
            x, y = next(self.train_loader)
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            logits_main, _ = model(x)
            loss = self.criterion(logits_main, y)
            loss.backward()

            self._clip_grad_norm(model)
            optimizer.step()
            prec1, prec5 = accuracy(logits_main, y, topk=(1, 5))
            metrics = {"prec1": prec1, "prec5": prec5, "loss": loss}
            metrics = reduce_metrics(metrics, self.distributed)
            meters.update(metrics)
            if self.main_proc and (step % self.log_frequency == 0 or step + 1 == self.steps_per_epoch):
                self.logger.info("Epoch [%d/%d] Step [%d/%d] (%s)  %s", epoch + 1, self.epochs,
                                 step + 1, self.steps_per_epoch, phase, meters)

    def _clip_grad_norm(self, model):
        if isinstance(model, DistributedDataParallel):
            nn.utils.clip_grad_norm_(model.module.parameters(), self.grad_clip)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

    def _reset_nan(self, parameters):
        with torch.no_grad():
            for param in parameters:
                for i, p in enumerate(param):
                    if p != p:  # equivalent to `isnan(p)`
                        param[i] = float("-inf")

    def _joint_train(self, epoch):
        self.model_large.train()
        self.model_small.train()
        meters = AverageMeterGroup()
        for step in range(self.steps_per_epoch):
            trn_x, trn_y = next(self.train_loader)
            val_x, val_y = next(self.valid_loader)
            trn_x, trn_y = trn_x.cuda(), trn_y.cuda()
            val_x, val_y = val_x.cuda(), val_y.cuda()

            # step 1. optimize architecture
            self.optimizer_alpha.zero_grad()
            self.optimizer_large.zero_grad()
            reg_decay = max(self.regular_coeff * (1 - float(epoch - self.warmup_epochs) / (
                (self.epochs - self.warmup_epochs) * self.regular_ratio)), 0)
            loss_regular = self.mutator_small.reset_with_loss()
            if loss_regular:
                loss_regular *= reg_decay
            logits_search, emsemble_logits_search = self.model_small(val_x)
            logits_main, emsemble_logits_main = self.model_large(val_x)
            loss_cls = (self.criterion(logits_search, val_y) + self.criterion(logits_main, val_y)) / self.loss_alpha
            loss_interactive = self.interactive_loss(emsemble_logits_search, emsemble_logits_main) * (self.loss_T ** 2) * self.loss_alpha
            loss = loss_cls + loss_interactive + loss_regular
            loss.backward()
            self._clip_grad_norm(self.model_large)
            self.optimizer_large.step()
            self.optimizer_alpha.step()
            # NOTE: need to call here `self._reset_nan(self.mutator_small.parameters())` if `cut_choices`

            # step 2. optimize op weights
            self.optimizer_small.zero_grad()
            with torch.no_grad():
                # resample architecture since parameters have been changed
                self.mutator_small.reset_with_loss()
            logits_search_train, _ = self.model_small(trn_x)
            loss_weight = self.criterion(logits_search_train, trn_y)
            loss_weight.backward()
            self._clip_grad_norm(self.model_small)
            self.optimizer_small.step()

            metrics = {"loss_cls": loss_cls, "loss_interactive": loss_interactive,
                       "loss_regular": loss_regular, "loss_weight": loss_weight}
            metrics = reduce_metrics(metrics, self.distributed)
            meters.update(metrics)

            if self.main_proc and (step % self.log_frequency == 0 or step + 1 == self.steps_per_epoch):
                self.logger.info("Epoch [%d/%d] Step [%d/%d] (joint)  %s", epoch + 1, self.epochs,
                                 step + 1, self.steps_per_epoch, meters)

    def train(self):
        for epoch in range(self.epochs):
            if epoch < self.warmup_epochs:
                with torch.no_grad():  # otherwise grads will be retained on the architecture params
                    self.mutator_small.reset_with_loss()
                self._warmup(PHASE_SMALL, epoch)
            else:
                with torch.no_grad():
                    self.mutator_large.reset()
                self._warmup(PHASE_LARGE, epoch)
                self._joint_train(epoch)

            self.export(os.path.join(self.checkpoint_dir, "epoch_{:02d}.json".format(epoch)),
                        os.path.join(self.checkpoint_dir, "epoch_{:02d}.genotypes".format(epoch)))

    def export(self, file, genotype_file):
        if self.main_proc:
            mutator_export, genotypes = self.mutator_small.export(self.logger)
            with open(file, "w") as f:
                json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)
            with open(genotype_file, "w") as f:
                f.write(str(genotypes))
