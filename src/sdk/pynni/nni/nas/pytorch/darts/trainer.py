import copy

import torch
from torch import nn as nn

from nni.nas.pytorch.trainer import Trainer
from nni.nas.utils import AverageMeterGroup, auto_device
from .mutator import DartsMutator


class DartsTrainer(Trainer):
    def __init__(self, model, loss, metrics,
                 model_optim, lr_scheduler, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.mutator = mutator
        if self.mutator is None:
            self.mutator = DartsMutator(model)
        self.model_optim = model_optim
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.device = auto_device() if device is None else device
        self.log_frequency = log_frequency

        self.model.to(self.device)
        self.loss.to(self.device)
        self.mutator.to(self.device)

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

    def train_epoch(self, epoch):
        self.model.train()
        self.mutator.train()
        lr = self.lr_scheduler.get_lr()[0]
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # backup model for hessian
            backup_model = copy.deepcopy(self.model.state_dict())
            # cannot deepcopy model because it will break the reference

            # phase 1. child network step
            self.model_optim.zero_grad()
            logits = self.model(trn_X)
            loss = self.loss(logits, trn_y)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.model_optim.step()

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
                print("Epoch {} Step [{}/{}]  {}".format(epoch, step, len(self.train_loader), meters))

        self.lr_scheduler.step()

    def validate_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (X, y) in enumerate(self.valid_loader):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                metrics = self.metrics(logits, y)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    print("Epoch {} Step [{}/{}]  {}".format(epoch, step, len(self.valid_loader), meters))

    def train(self):
        for epoch in range(self.num_epochs):
            # training
            print("Epoch {} Training".format(epoch))
            self.train_epoch(epoch)

            # validation
            print("Epoch {} Validating".format(epoch))
            self.validate_epoch(epoch)

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y, backup_model, lr):
        """
        Compute unrolled loss and backward its gradients
        Parameters
        ----------
        v_model: backup model before this step
        lr: learning rate for virtual gradient step (same as net lr)
        """
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

            loss = self.loss(self.model(trn_X), trn_y)  # TODO: should use model instead of self.model
            if e > 0:
                dalpha_pos = torch.autograd.grad(loss, self.mutator.parameters())  # dalpha { L_trn(w+) }
            elif e < 0:
                dalpha_neg = torch.autograd.grad(loss, self.mutator.parameters())  # dalpha { L_trn(w-) }

        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def finalize(self):
        pass
