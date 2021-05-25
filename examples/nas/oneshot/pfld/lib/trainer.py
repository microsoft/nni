# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import os
import time
import torch

import numpy as np

from nni.algorithms.nas.pytorch.fbnet import FBNetTrainer
from nni.nas.pytorch.utils import AverageMeter
from .utils import accuracy


class PFLDTrainer(FBNetTrainer):
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

        super(PFLDTrainer, self).__init__(
            model,
            model_optim,
            criterion,
            device,
            device_ids,
            lookup_table,
            train_loader,
            valid_loader,
            n_epochs,
            load_ckpt,
            arch_path,
            logger,
        )

        # DataParallel of the AuxiliaryNet to PFLD
        self.auxiliarynet = auxiliarynet
        self.auxiliarynet = torch.nn.DataParallel(
            self.auxiliarynet, device_ids=device_ids
        )
        self.auxiliarynet.to(device)

    def _validate(self):
        """
        Do validation. During validation, LayerChoices use the mixed-op.

        Returns
        -------
        float, float
            average loss, average nme
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

            lands, feats = self.model(img)
            landmarks = lands.squeeze()
            angle = self.auxiliarynet(feats)

            # task loss
            weighted_loss, l2_loss = self.criterion(
                landmark_gt, angle_gt, angle, landmarks
            )
            loss = l2_loss if arch_train else weighted_loss

            # hardware-aware loss
            perf_cost = self._get_perf_cost(requires_grad=True)
            regu_loss = self.reg_loss(perf_cost)
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
        arch_param_num = self.mutator.num_arch_params()
        self.logger.info("#arch_params: {}".format(arch_param_num))
        self.epoch = max(self.start_epoch, self.epoch)

        ckpt_path = self.config.model_dir
        choice_names = None
        val_nme = 1e6

        for epoch in range(self.epoch, self.n_epochs):
            # update the weight parameters
            self.logger.info("\n--------Train epoch: %d--------\n", epoch + 1)
            self._train_epoch(epoch, self.model_optim)
            # adjust learning rate
            self.scheduler.step()

            # update the architecture parameters
            self.logger.info("Update architecture parameters")
            self.mutator.arch_requires_grad()
            self._train_epoch(epoch, self.arch_optimizer, True)
            self.mutator.arch_disable_grad()
            # temperature annealing
            self.temp = self.temp * self.exp_anneal_rate
            self.mutator.set_temperature(self.temp)
            # sample the architecture of sub-network
            choice_names = self._layer_choice_sample()

            # validate
            _, nme = self._validate()

            if epoch % 10 == 0:
                filename = os.path.join(ckpt_path, "checkpoint_%s.pth" % epoch)
                self.save_checkpoint(epoch, filename, choice_names)

            if nme < val_nme:
                filename = os.path.join(ckpt_path, "checkpoint_best.pth")
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
