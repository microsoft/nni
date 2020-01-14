import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import apex
from apex.parallel import DistributedDataParallel
from nni.nas.pytorch.darts import DartsMutator
from nni.nas.pytorch.mutables import LayerChoice
from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.utils import AverageMeterGroup
from utils import CyclicIterator, TorchTensorEncoder, accuracy, reduce_metrics

PHASE_SMALL = "small"
PHASE_LARGE = "large"


class RegularizedDartsMutator(DartsMutator):
    def reset(self):
        raise ValueError("You should probably call `reset_with_loss`.")

    def cut_choices(self, cut_num=2):
        # `cut_choices` is implemented but not used
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                _, idx = torch.topk(-self.choices[mutable.key], cut_num)
                with torch.no_grad():
                    for i in idx:
                        self.choices[mutable.key][i] = -float("inf")

    def reset_with_loss(self):
        self._cache, reg_loss = self.sample_search()
        return reg_loss

    def sample_search(self):
        result = super().sample_search()
        loss = []
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                def need_reg(choice):
                    return any(t in str(type(choice)).lower() for t in ["poolwithoutbn", "identity", "dilconv"])

                for i, choice in enumerate(mutable.choices):
                    if need_reg(choice):
                        norm = torch.abs(self.choices[mutable.key][i])
                        if norm < 1E10:
                            loss.append(norm)
        if not loss:
            return result, None
        return result, sum(loss)

    def export(self, logger):
        result = self.sample_final()
        if hasattr(self.model, "plot_genotype"):
            genotypes = self.model.plot_genotype(result, logger)
        return result, genotypes


class RegularizedMutatorParallel(DistributedDataParallel):
    def reset_with_loss(self):
        result = self.module.reset_with_loss()
        self.callback_queued = False
        return result

    def cut_choices(self, *args, **kwargs):
        self.module.cut_choices(*args, **kwargs)

    def export(self, logger):
        return self.module.export(logger)


class DartsDiscreteMutator(Mutator):

    def __init__(self, model, parent_mutator):
        super().__init__(model)
        self.__dict__["parent_mutator"] = parent_mutator  # avoid parameters to be included

    def sample_search(self):
        return self.parent_mutator.sample_final()


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
    def __init__(self, model_small, model_large, criterion, loaders, samplers, logger, config):
        train_loader, valid_loader = loaders
        train_sampler, valid_sampler = samplers
        self.train_loader = CyclicIterator(train_loader, train_sampler, config.distributed)
        self.valid_loader = CyclicIterator(valid_loader, valid_sampler, config.distributed)

        self.regular_coeff = config.regular_coeff
        self.regular_ratio = config.regular_ratio
        self.warmup_epochs = config.warmup_epochs
        self.fix_head = config.fix_head
        self.epochs = config.epochs
        self.steps_per_epoch = config.steps_per_epoch
        if self.steps_per_epoch is None:
            self.steps_per_epoch = min(len(self.train_loader), len(self.valid_loader))
        self.loss_alpha = config.loss_alpha
        self.grad_clip = config.grad_clip
        if config.interactive_type == "kl":
            self.interactive_loss = InteractiveKLLoss(config.loss_T)
        elif config.interactive_type == "smoothl1":
            self.interactive_loss = nn.SmoothL1Loss()
        self.loss_T = config.loss_T
        self.distributed = config.distributed
        self.log_frequency = config.log_frequency
        self.main_proc = not config.distributed or config.local_rank == 0

        self.logger = logger
        self.checkpoint_dir = config.output_path
        if self.main_proc:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if config.distributed:
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

        self.optimizer_small = torch.optim.SGD(self.model_small.parameters(), config.w_lr,
                                               momentum=config.w_momentum, weight_decay=config.w_weight_decay)
        self.optimizer_large = torch.optim.SGD(self.model_large.parameters(), config.nasnet_lr,
                                               momentum=config.w_momentum, weight_decay=config.w_weight_decay)
        self.optimizer_alpha = torch.optim.Adam(self.mutator_small.parameters(), config.alpha_lr,
                                                betas=(0.5, 0.999), weight_decay=config.alpha_weight_decay)

        if config.distributed:
            apex.parallel.convert_syncbn_model(self.model_small)
            apex.parallel.convert_syncbn_model(self.model_large)
            self.model_small = DistributedDataParallel(self.model_small, delay_allreduce=True)
            self.model_large = DistributedDataParallel(self.model_large, delay_allreduce=True)
            self.mutator_small = RegularizedMutatorParallel(self.mutator_small, delay_allreduce=True)
            if config.share_module:
                self.model_small.callback_queued = True
                self.model_large.callback_queued = True
            # mutator large never gets optimized, so do not need parallelized

    def warmup(self, phase, epoch):
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

    def joint_train(self, epoch):
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
                self.warmup(PHASE_SMALL, epoch)
            else:
                with torch.no_grad():
                    self.mutator_large.reset()
                self.warmup(PHASE_LARGE, epoch)
                self.joint_train(epoch)

            self.export(os.path.join(self.checkpoint_dir, "epoch_{:02d}.json".format(epoch)),
                        os.path.join(self.checkpoint_dir, "epoch_{:02d}.genotypes".format(epoch)))

    def export(self, file, genotype_file):
        if self.main_proc:
            mutator_export, genotypes = self.mutator_small.export(self.logger)
            with open(file, "w") as f:
                json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)
            with open(genotype_file, "w") as f:
                f.write(str(genotypes))
