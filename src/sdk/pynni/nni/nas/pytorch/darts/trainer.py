import copy
import logging

import torch
from torch import nn as nn

from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .mutator import DartsMutator

logger = logging.getLogger("darts/trainer")


class DartsTrainer(Trainer):
    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None):
        super().__init__(model, mutator if mutator is not None else DartsMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)
        self.ctrl_optim = torch.optim.Adam(self.mutator.parameters(), 3.0E-4, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        n_train = len(self.dataset_train)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=batch_size,
                                                       num_workers=workers)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.mutator.train()
        lr = self.optimizer.param_groups[0]["lr"]
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # backup model for hessian
            backup_model = copy.deepcopy(self.model.state_dict())
            # cannot deepcopy model because it will break the reference

            # phase 1. child network step
            self.optimizer.zero_grad()
            self.mutator.reset()
            logits = self.model(trn_X)
            loss = self.loss(logits, trn_y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()

            new_model = copy.deepcopy(self.model.state_dict())

            # phase 2. architect step (alpha)
            self.ctrl_optim.zero_grad()
            # compute unrolled loss
            self._unrolled_backward(trn_X, trn_y, val_X, val_y, backup_model, lr)
            self.ctrl_optim.step()

            self.model.load_state_dict(new_model)

            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Epoch [{}/{}] Step [{}/{}]  {}".format(epoch,
                                                                    self.num_epochs, step, len(self.train_loader), meters))

    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            self.mutator.reset()
            for step, (X, y) in enumerate(self.test_loader):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                metrics = self.metrics(logits, y)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [{}/{}] Step [{}/{}]  {}".format(epoch,
                                                                        self.num_epochs, step, len(self.valid_loader), meters))

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y, backup_model, lr):
        """
        Compute unrolled loss and backward its gradients
        Parameters
        ----------
        v_model: backup model before this step
        lr: learning rate for virtual gradient step (same as net lr)
        """
        self.mutator.reset()
        loss = self.loss(self.model(val_X), val_y)
        w_model = tuple(self.model.parameters())
        w_ctrl = tuple(self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model = w_grads[:len(w_model)]
        d_ctrl = w_grads[len(w_model):]

        hessian = self._compute_hessian(backup_model, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                param.grad = d - lr * h

    def _compute_hessian(self, model, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        self.model.load_state_dict(model)

        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    p += eps * d

            self.mutator.reset()
            loss = self.loss(self.model(trn_X), trn_y)
            if e > 0:
                dalpha_pos = torch.autograd.grad(loss, self.mutator.parameters())  # dalpha { L_trn(w+) }
            elif e < 0:
                dalpha_neg = torch.autograd.grad(loss, self.mutator.parameters())  # dalpha { L_trn(w-) }

        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
