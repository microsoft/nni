# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import json
import os
import time
import torch

import numpy as np

from torch.autograd import Variable
from nni.nas.pytorch.base_trainer import BaseTrainer
from nni.nas.pytorch.trainer import TorchTensorEncoder
from nni.nas.pytorch.utils import AverageMeter
from .mutator import FBNetMutator
from .utils import RegularizerLoss, accuracy


class FBNetTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        model_optim,
        criterion,
        device,
        device_ids,
        lookup_table,
        train_loader,
        valid_loader,
        n_epochs=120,
        load_ckpt=False,
        arch_path=None,
        logger=None,
    ):
        """
        Parameters
        ----------
        model : pytorch model
            the user model, which has mutables
        model_optim : pytorch optimizer
            the user defined optimizer
        criterion : pytorch loss
            the main task loss, nn.CrossEntropyLoss() is for classification
        device : pytorch device
            the devices to train/search the model
        device_ids : list of int
            the indexes of devices used for training
        lookup_table : class
            lookup table object for fbnet training
        train_loader : pytorch data loader
            data loader for the training set
        valid_loader : pytorch data loader
            data loader for the validation set
        n_epochs : int
            number of epochs to train/search
        load_ckpt : bool
            whether load checkpoint
        arch_path : str
            the path to store chosen architecture
        logger : logger
            the logger
        """
        self.model = model
        self.model_optim = model_optim
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.dev_num = len(device_ids)
        self.n_epochs = n_epochs
        self.lookup_table = lookup_table
        self.config = lookup_table.config
        self.start_epoch = self.config.start_epoch
        self.temp = self.config.init_temperature
        self.exp_anneal_rate = self.config.exp_anneal_rate
        self.mode = self.config.mode

        self.load_ckpt = load_ckpt
        self.arch_path = arch_path
        self.logger = logger

        # scheduler of learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=n_epochs, last_epoch=-1
        )

        # init mutator
        self.mutator = FBNetMutator(model, lookup_table)
        self.mutator.set_temperature(self.temp)

        # DataParallel should be put behind the init of mutator
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device)

        # build architecture optimizer
        self.arch_optimizer = torch.optim.AdamW(
            self.mutator.get_architecture_parameters(),
            self.config.nas_lr,
            weight_decay=self.config.nas_weight_decay,
        )
        self.reg_loss = RegularizerLoss(config=self.config)

        self.criterion = criterion
        self.epoch = 0

    def _layer_choice_sample(self):
        """
        Sample the index of network within layer choice
        """
        stages = [stage_name for stage_name in self.lookup_table.layer_num]
        stage_lnum = [self.lookup_table.layer_num[stage] for stage in stages]

        # get the choice idx in each layer
        choice_ids = list()
        layer_id = 0
        for param in self.mutator.get_architecture_parameters():
            param_np = param.cpu().detach().numpy()
            op_idx = np.argmax(param_np)
            choice_ids.append(op_idx)
            self.logger.info(
                "layer {}: {}, index: {}".format(layer_id, param_np, op_idx)
            )
            layer_id += 1

        # get the arch_sample
        choice_names = list()
        layer_id = 0
        for i, stage_name in enumerate(stages):
            ops_names = [op for op in self.lookup_table.lut_ops[stage_name]]
            for j in range(stage_lnum[i]):
                searched_op = ops_names[choice_ids[layer_id]]
                choice_names.append(searched_op)
                layer_id += 1

        self.logger.info(choice_names)
        return choice_names

    def _get_perf_cost(self, requires_grad=True):
        """
        Get the accumulated performance cost.
        """
        perf_cost = Variable(
            torch.zeros(1), requires_grad=requires_grad
        ).to(self.device, non_blocking=True)

        for latency in self.mutator.get_weighted_latency():
            perf_cost = perf_cost + latency

        return perf_cost

    def _validate(self):
        """
        Do validation. During validation, LayerChoices use the mixed-op.

        Returns
        -------
        float, float, float
            average loss, average top1 accuracy, average top5 accuracy
        """
        self.valid_loader.batch_sampler.drop_last = False
        batch_time = AverageMeter("batch_time")
        losses = AverageMeter("losses")
        top1 = AverageMeter("top1")
        top5 = AverageMeter("top5")

        # test on validation set under eval mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.valid_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                output = self.model(images)

                loss = self.criterion(output, labels)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(self.valid_loader):
                    test_log = (
                        "Valid" + ": [{0}/{1}]\t"
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Top-5 acc {top5.val:.3f} ({top5.avg:.3f})".format(
                            i,
                            len(self.valid_loader) - 1,
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1,
                            top5=top5,
                        )
                    )
                    self.logger.info(test_log)

        return losses.avg, top1.avg, top5.avg

    def _train_epoch(self, epoch, optimizer, arch_train=False):
        """
        Train one epoch.
        """
        batch_time = AverageMeter("batch_time")
        data_time = AverageMeter("data_time")
        losses = AverageMeter("losses")
        top1 = AverageMeter("top1")
        top5 = AverageMeter("top5")

        # switch to train mode
        self.model.train()

        data_loader = self.valid_loader if arch_train else self.train_loader
        end = time.time()
        for i, (images, labels) in enumerate(data_loader):
            data_time.update(time.time() - end)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(images)
            loss = self.criterion(output, labels)

            # hardware-aware loss
            perf_cost = self._get_perf_cost(requires_grad=True)
            regu_loss = self.reg_loss(perf_cost)
            if self.mode.startswith("mul"):
                loss = loss * regu_loss
            elif self.mode.startswith("add"):
                loss = loss + regu_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                batch_log = (
                    "Warmup Train [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    "Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\t".format(
                        epoch + 1,
                        i,
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                self.logger.info(batch_log)

    def _warm_up(self):
        """
        Warm up the model, while the architecture weights are not trained.
        """
        for epoch in range(self.epoch, self.start_epoch):
            self.logger.info("\n--------Warmup epoch: %d--------\n", epoch + 1)
            self._train_epoch(epoch, self.model_optim)
            # adjust learning rate
            self.scheduler.step()

            # validation
            val_loss, val_top1, val_top5 = self._validate()
            val_log = (
                "Warmup Valid [{0}/{1}]\t"
                "loss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}".format(
                    epoch + 1, self.warmup_epochs, val_loss, val_top1, val_top5
                )
            )
            self.logger.info(val_log)

            if epoch % 10 == 0:
                filename = os.path.join(
                    self.config.model_dir, "checkpoint_%s.pth" % epoch
                )
                self.save_checkpoint(epoch, filename)

    def _train(self):
        """
        Train the model, it trains model weights and architecute weights.
        Architecture weights are trained according to the schedule.
        Before updating architecture weights, ```requires_grad``` is enabled.
        Then, it is disabled after the updating, in order not to update
        architecture weights when training model weights.
        """
        arch_param_num = self.mutator.num_arch_params()
        self.logger.info("#arch_params: {}".format(arch_param_num))
        self.epoch = max(self.start_epoch, self.epoch)

        ckpt_path = self.config.model_dir
        choice_names = None
        top1_best = 0.0

        for epoch in range(self.epoch, self.n_epochs):
            self.logger.info("\n--------Train epoch: %d--------\n", epoch + 1)
            # update the weight parameters
            self._train_epoch(epoch, self.model_optim)
            # adjust learning rate
            self.scheduler.step()

            self.logger.info("Update architecture parameters")
            # update the architecture parameters
            self.mutator.arch_requires_grad()
            self._train_epoch(epoch, self.arch_optimizer, True)
            self.mutator.arch_disable_grad()
            # temperature annealing
            self.temp = self.temp * self.exp_anneal_rate
            self.mutator.set_temperature(self.temp)
            # sample the architecture of sub-network
            choice_names = self._layer_choice_sample()

            # validate
            val_loss, val_top1, val_top5 = self._validate()
            val_log = (
                "Valid [{0}]\t"
                "loss {1:.3f}\ttop-1 acc {2:.3f} \ttop-5 acc {3:.3f}".format(
                    epoch + 1, val_loss, val_top1, val_top5
                )
            )
            self.logger.info(val_log)

            if epoch % 10 == 0:
                filename = os.path.join(ckpt_path, "checkpoint_%s.pth" % epoch)
                self.save_checkpoint(epoch, filename, choice_names)

            val_top1 = val_top1.cpu().as_numpy()
            if val_top1 > top1_best:
                filename = os.path.join(ckpt_path, "checkpoint_best.pth")
                self.save_checkpoint(epoch, filename, choice_names)
                top1_best = val_top1

    def save_checkpoint(self, epoch, filename, choice_names=None):
        """
        Save checkpoint of the whole model.
        Saving model weights and architecture weights as ```filename```,
        and saving currently chosen architecture in ```arch_path```.
        """
        state = {
            "model": self.model.state_dict(),
            "optim": self.model_optim.state_dict(),
            "epoch": epoch,
            "arch_sample": choice_names,
        }
        torch.save(state, filename)
        self.logger.info("Save checkpoint to {0:}".format(filename))

        if self.arch_path:
            self.export(self.arch_path)

    def load_checkpoint(self, filename):
        """
        Load the checkpoint from ```ckpt_path```.
        """
        ckpt = torch.load(filename)
        self.epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.model_optim.load_state_dict(ckpt["optim"])

    def train(self):
        """
        Train the whole model.
        """
        if self.load_ckpt:
            ckpt_path = self.config.model_dir
            filename = os.path.join(ckpt_path, "checkpoint_best.pth")
            if os.path.exists(filename):
                self.load_checkpoint(filename)

        if self.epoch < self.start_epoch:
            self._warm_up()
        self._train()

    def export(self, file_name):
        """
        Export the chosen architecture into a file

        Parameters
        ----------
        file_name : str
            the file that stores exported chosen architecture
        """
        exported_arch = self.mutator.sample_final()
        with open(file_name, "w") as f:
            json.dump(
                exported_arch,
                f,
                indent=2,
                sort_keys=True,
                cls=TorchTensorEncoder,
            )

    def validate(self):
        raise NotImplementedError

    def checkpoint(self):
        raise NotImplementedError
