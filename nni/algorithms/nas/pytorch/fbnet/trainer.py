# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import time
import json
import os
import torch

import numpy as np

from torch import nn as nn
from torch.autograd import Variable
from nni.nas.pytorch.base_trainer import BaseTrainer
from nni.nas.pytorch.trainer import TorchTensorEncoder
from nni.nas.pytorch.utils import AverageMeter
from nni.algorithms.nas.pytorch.fbnet import FBNetMutator
from .utils import accuracy


class RegularizerLoss(nn.Module):
    """Auxilliary loss for hardware-aware NAS."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : class
            to manage the configuration for NAS training, and search space etc.
        """
        super(RegularizerLoss, self).__init__()
        self.mode = config.mode
        self.alpha = config.alpha
        self.beta = config.beta

    def forward(self, perf_cost, batch_size=1):
        """
        Parameters
        ----------
        perf_cost : tensor
            the accumulated performance cost
        batch_size : int
            batch size for normalization

        Returns
        -------
        output: tensor
            the hardware-aware constraint loss
        """
        if self.mode == "mul":
            log_loss = torch.log(perf_cost / batch_size) ** self.beta
            return self.alpha * log_loss
        elif self.mode == "add":
            linear_loss = (perf_cost / batch_size) ** self.beta
            return self.alpha * linear_loss
        else:
            raise NotImplementedError


class FBNetTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        auxiliarynet,
        model_optim,
        criterion,
        device,
        device_ids,
        config,
        lookup_table,
        train_loader,
        valid_loader,
        n_epochs=300,
        load_ckpt=False,
        arch_path=None,
        logger=None,
    ):
        """
        Parameters
        ----------
        model : pytorch model
            the user model, which has mutables
        auxiliarynet : pytorch model
            the auxiliarynet to regress angle
        model_optim : pytorch optimizer
            the user defined optimizer
        criterion : pytorch loss
            the main task loss
        device : pytorch device
            the devices to train/search the model
        device_ids : list of int
            the indexes of devices used for training
        config : class
            configuration object for fbnet training
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
        self.auxiliarynet = auxiliarynet
        self.model_optim = model_optim
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.dev_num = len(device_ids)
        self.n_epochs = n_epochs
        self.config = config
        self.lookup_table = lookup_table
        self.arch_search = config.arch_search
        self.start_epoch = config.start_epoch
        self.temp = config.init_temperature
        self.exp_anneal_rate = config.exp_anneal_rate
        self.mode = config.mode

        self.load_ckpt = load_ckpt
        self.arch_path = arch_path
        self.logger = logger

        # scheduler of learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=n_epochs, last_epoch=-1
        )

        if self.arch_search:
            # init mutator
            self.mutator = FBNetMutator(model, lookup_table)

        # DataParallel should be put behind the init of mutator
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device)
        self.auxiliarynet = torch.nn.DataParallel(
            self.auxiliarynet, device_ids=device_ids
        )
        self.auxiliarynet.to(device)

        if self.arch_search:
            # build architecture optimizer
            self.arch_optimizer = torch.optim.AdamW(
                self.mutator.get_architecture_parameters(),
                config.nas_lr,
                weight_decay=config.nas_weight_decay,
            )
            self.reg_loss = RegularizerLoss(config=config)

        self.criterion = criterion
        self.epoch = 0

    def _layer_choice_sample(self):
        """
        sample the index of network within layer choice
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

    def _validate(self):
        """
        Do validation. During validation, LayerChoices use the mixed-op.

        Returns
        -------
        float, float, float
            average loss, average top1 accuracy, average top5 accuracy
        """

        # test on validation set under eval mode
        self.model.eval()
        self.auxiliarynet.eval()

        losses, nme = list(), list()
        batch_time = AverageMeter("batch_time")
        end = time.time()
        with torch.no_grad():
            for i, (img, land_gt, angle_gt) in enumerate(self.valid_loader):
                img = img.to(self.device, non_blocking=True)
                landmark_gt = land_gt.to(self.device, non_blocking=True)
                angle_gt = angle_gt.to(self.device, non_blocking=True)

                if self.arch_search:
                    perf_cost = Variable(torch.zeros(self.dev_num, 1)).to(
                        self.device, non_blocking=True
                    )
                    landmark, _, _ = self.model(img, self.temp, perf_cost)

                else:
                    landmark, _ = self.model(img)

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

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        self.logger.info("===> Evaluate:")
        self.logger.info(
            "Eval set: Average loss: {:.4f} nme: {:.4f}".format(
                np.mean(losses), np.mean(nme)
            )
        )
        return np.mean(losses), np.mean(nme)

    def _train_epoch(self, epoch, optimizer, arch_train=False):
        """
        Train one epoch.
        """
        # switch to train mode
        self.model.train()
        self.auxiliarynet.train()

        batch_time = AverageMeter("batch_time")
        data_time = AverageMeter("data_time")
        losses = AverageMeter("losses")

        data_loader = self.valid_loader if arch_train else self.train_loader
        end = time.time()
        for i, (img, landmark_gt, angle_gt) in enumerate(data_loader):
            data_time.update(time.time() - end)
            img = img.to(self.device, non_blocking=True)
            landmark_gt = landmark_gt.to(self.device, non_blocking=True)
            angle_gt = angle_gt.to(self.device, non_blocking=True)

            if self.arch_search:
                perf_cost = Variable(
                    torch.zeros(self.dev_num, 1), requires_grad=True
                ).to(self.device, non_blocking=True)
                lands, feats, perf_cost = self.model(img, self.temp, perf_cost)
            else:
                lands, feats = self.model(img)
            landmarks = lands.squeeze()
            angle = self.auxiliarynet(feats)

            # task loss
            weighted_loss, l2_loss = self.criterion(
                landmark_gt, angle_gt, angle, landmarks
            )
            loss = l2_loss if arch_train else weighted_loss

            if self.arch_search:
                # hardware-aware loss
                regu_loss = self.reg_loss(perf_cost.mean(dim=0))
                if self.mode.startswith("mul"):
                    loss = loss * regu_loss
                elif self.mode.startswith("add"):
                    loss = loss + regu_loss

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
            _, _ = self._validate()
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
        if self.arch_search:
            arch_param_num = self.mutator.num_arch_params()
            self.logger.info("#arch_params: {}".format(arch_param_num))
            self.epoch = max(self.start_epoch, self.epoch)

        ckpt_path = self.config.model_dir
        choice_names = None
        val_nme = 1e6

        for epoch in range(self.epoch, self.n_epochs):
            self.logger.info("\n--------Train epoch: %d--------\n", epoch + 1)
            # update the weight parameters
            self._train_epoch(epoch, self.model_optim)
            # adjust learning rate
            self.scheduler.step()

            if self.arch_search:
                self.logger.info("Update architecture parameters")
                # update the architecture parameters
                self.mutator.arch_requires_grad()
                self._train_epoch(epoch, self.arch_optimizer, True)
                self.mutator.arch_disable_grad()
                # temperature annealing
                self.temp = self.temp * self.exp_anneal_rate
                # sample the architecture of sub-network
                choice_names = self._layer_choice_sample()

            # validate
            _, nme = self._validate()

            if epoch % 10 == 0:
                filename = os.path.join(ckpt_path, "checkpoint_%s.pth" % epoch)
                self.save_checkpoint(epoch, filename, choice_names)

            if nme < val_nme:
                filename = os.path.join(ckpt_path, "checkpoint_min_nme.pth")
                self.save_checkpoint(epoch, filename, choice_names)
                val_nme = nme
            self.logger.info("Best nme: {:.4f}".format(val_nme))

    def save_checkpoint(self, epoch, filename, choice_names=None):
        """
        Save checkpoint of the whole model.
        Saving model weights and architecture weights as ```filename```,
        and saving currently chosen architecture in ```arch_path```.
        """
        state = {
            "pfld_backbone": self.model.state_dict(),
            "auxiliarynet": self.auxiliarynet.state_dict(),
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
        Load the checkpoint from ```filename```.
        """
        ckpt = torch.load(filename)
        self.epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["pfld_backbone"])
        self.auxiliarynet.load_state_dict(ckpt["auxiliarynet"])
        self.model_optim.load_state_dict(ckpt["optim"])

    def train(self):
        """
        Train the whole model.
        """
        if self.load_ckpt:
            ckpt_path = self.config.model_dir
            filename = os.path.join(ckpt_path, "checkpoint_min_nme.pth")
            if os.path.exists(filename):
                self.load_checkpoint(filename)

        if (self.epoch < self.start_epoch) and self.arch_search:
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
