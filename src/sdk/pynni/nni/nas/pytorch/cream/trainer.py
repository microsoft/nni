# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

logger = logging.getLogger(__name__)


class CreamSupernetTrainer(Trainer):
    """
    This trainer trains a supernet that can be used for evolution search.

    Parameters
    ----------
    model : nn.Module
        Model with mutables.
    mutator : Mutator
        A mutator object that has been initialized with the model.
    loss : callable
        Called with logits and targets. Returns a loss tensor.
    metrics : callable
        Returns a dict that maps metrics keys to metrics data.
    optimizer : Optimizer
        Optimizer that optimizes the model.
    num_epochs : int
        Number of epochs of training.
    train_loader : iterable
        Data loader of training. Raise ``StopIteration`` when one epoch is exhausted.
    dataset_valid : iterable
        Data loader of validation. Raise ``StopIteration`` when one epoch is exhausted.
    batch_size : int
        Batch size.
    workers: int
        Number of threads for data preprocessing. Not used for this trainer. Maybe removed in future.
    device : torch.device
        Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
        automatic detects GPU and selects GPU first.
    log_frequency : int
        Number of mini-batches to log metrics.
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.
    """

    def __init__(self, model, loss,
                 optimizer, num_epochs, train_loader, valid_loader,
                 mutator=None, batch_size=64, log_frequency=None,
                 est=None, meta_sta_epoch=20, update_iter=200, slices=2, pool_size=10,
                 pick_method='meta', lr_scheduler=None, distributed=True, local_rank=0, val_loss=None):
        assert torch.cuda.is_available()
        super(CreamSupernetTrainer, self).__init__(model, mutator, loss, None, optimizer, num_epochs,
                                                   train_loader, valid_loader, batch_size, 8,
                                                   'cuda', log_frequency, None)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.mutator = mutator
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.est = est
        self.best_children_pool = []
        self.num_epochs = num_epochs
        self.meta_sta_epoch = meta_sta_epoch
        self.update_iter = update_iter
        self.slices = slices
        self.pick_method = pick_method
        self.pool_size = pool_size
        self.main_proc = not distributed or local_rank == 0
        self.distributed = distributed
        self.val_loss = val_loss
        self.lr_scheduler = lr_scheduler
        self.callbacks = []
        self.arch_dict = dict()

    def cross_entropy_loss_with_soft_target(self, pred, soft_target):
        logsoftmax = nn.LogSoftmax()
        return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= float(os.environ["WORLD_SIZE"])
        return rt

    def reduce_metrics(self, metrics, distributed=False):
        if distributed:
            return {k: self.reduce_tensor(v).item() for k, v in metrics.items()}
        return {k: v.item() for k, v in metrics.items()}

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]

    def train_one_epoch(self, epoch):
        def get_model(model):
            return model.module

        meters = AverageMeterGroup()
        for step, (input_data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            self.mutator.reset()

            input_data = input_data.cuda()
            target = target.cuda()

            cand_flops = self.est.get_flops(self.mutator._cache)

            if epoch > self.meta_sta_epoch and step > 0 and step % self.update_iter == 0:

                slice_ind = self.slices
                x = deepcopy(input_data[:slice_ind].clone().detach())

                if self.best_children_pool:
                    if self.pick_method == 'top1':
                        meta_value, cand = 1, sorted(self.best_children_pool, reverse=True)[0][3]
                    elif self.pick_method == 'meta':
                        meta_value, cand_idx, cand = -1000000000, -1, None
                        for now_idx, item in enumerate(self.best_children_pool):
                            inputx = item[3]
                            output = F.softmax(self.model(inputx), dim=1)
                            weight = get_model(self.model).forward_meta(output - item[4])
                            if weight > meta_value:
                                meta_value = weight  # deepcopy(torch.nn.functional.sigmoid(weight))
                                cand_idx = now_idx
                                cand = self.arch_dict[(self.best_children_pool[cand_idx][0],
                                                       self.best_children_pool[cand_idx][2])]
                        assert cand is not None
                        meta_value = torch.nn.functional.sigmoid(-weight)
                    else:
                        raise ValueError('Method Not supported')

                    u_output = self.model(x)

                    saved_cache = self.mutator._cache
                    self.mutator._cache = cand
                    u_teacher_output = self.model(x)
                    self.mutator._cache = saved_cache

                    u_soft_label = F.softmax(u_teacher_output, dim=1)
                    kd_loss = meta_value * self.cross_entropy_loss_with_soft_target(u_output, u_soft_label)
                    self.optimizer.zero_grad()

                    grad_1 = torch.autograd.grad(kd_loss,
                                                 get_model(self.model).rand_parameters(self.mutator._cache),
                                                 create_graph=True)

                    def raw_sgd(w, g):
                        return g * self.optimizer.param_groups[-1]['lr'] + w

                    students_weight = [raw_sgd(p, grad_item)
                                       for p, grad_item in
                                       zip(get_model(self.model).rand_parameters(self.mutator._cache), grad_1)]

                    # update student weights
                    for weight, grad_item in zip(get_model(self.model).rand_parameters(self.mutator._cache), grad_1):
                        weight.grad = grad_item
                    torch.nn.utils.clip_grad_norm_(get_model(self.model).rand_parameters(self.mutator._cache), 1)
                    self.optimizer.step()
                    for weight, grad_item in zip(get_model(self.model).rand_parameters(self.mutator._cache), grad_1):
                        del weight.grad

                    held_out_x = input_data[slice_ind:slice_ind * 2].clone()
                    output_2 = self.model(held_out_x)
                    valid_loss = self.loss(output_2, target[slice_ind:slice_ind * 2])
                    self.optimizer.zero_grad()

                    grad_student_val = torch.autograd.grad(valid_loss,
                                                           get_model(self.model).rand_parameters(self.mutator._cache),
                                                           retain_graph=True)

                    grad_teacher = torch.autograd.grad(students_weight[0],
                                                       get_model(self.model).rand_parameters(cand,
                                                                                             self.pick_method == 'meta'),
                                                       grad_outputs=grad_student_val)

                    # update teacher model
                    for weight, grad_item in zip(
                            get_model(self.model).rand_parameters(cand, self.pick_method == 'meta'),
                            grad_teacher):
                        weight.grad = grad_item
                    torch.nn.utils.clip_grad_norm_(
                        get_model(self.model).rand_parameters(self.mutator._cache, self.pick_method == 'meta'), 1)
                    self.optimizer.step()
                    for weight, grad_item in zip(
                            get_model(self.model).rand_parameters(cand, self.pick_method == 'meta'),
                            grad_teacher):
                        del weight.grad

                    for item in students_weight:
                        del item
                    del grad_teacher, grad_1, grad_student_val, x, held_out_x
                    del valid_loss, kd_loss, u_soft_label, u_output, u_teacher_output, output_2

                else:
                    raise ValueError("Must 1nd or 2nd update teacher weights")

            # get_best_teacher
            if self.best_children_pool:
                if self.pick_method == 'top1':
                    meta_value, cand = 0.5, sorted(self.best_children_pool, reverse=True)[0][3]
                elif self.pick_method == 'meta':
                    meta_value, cand_idx, cand = -1000000000, -1, None
                    for now_idx, item in enumerate(self.best_children_pool):
                        inputx = item[3]
                        output = F.softmax(self.model(inputx), dim=1)
                        weight = get_model(self.model).forward_meta(output - item[4])
                        if weight > meta_value:
                            meta_value = weight
                            cand_idx = now_idx
                            cand = self.arch_dict[(self.best_children_pool[cand_idx][0],
                                                   self.best_children_pool[cand_idx][2])]
                    assert cand is not None
                    meta_value = torch.nn.functional.sigmoid(-weight)
                else:
                    raise ValueError('Method Not supported')

            if not self.best_children_pool:
                output = self.model(input)
                loss = self.loss(output, target)
                kd_loss = loss
            elif epoch <= self.meta_sta_epoch:
                output = self.model(input)
                loss = self.loss(output, target)
            else:
                output = self.model(input)
                with torch.no_grad():
                    # save student arch
                    saved_cache = self.mutator._cache
                    self.mutator._cache = cand

                    # forward
                    teacher_output = self.model(input).detach()

                    # restore student arch
                    self.mutator._cache = saved_cache
                    soft_label = F.softmax(teacher_output, dim=1)
                kd_loss = self.cross_entropy_loss_with_soft_target(output, soft_label)
                valid_loss = self.loss(output, target)
                loss = (meta_value * kd_loss + (2 - meta_value) * valid_loss) / 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec1, prec5 = self.accuracy(output, target, topk=(1, 5))
            metrics = {"prec1": prec1, "prec5": prec5, "loss": loss}
            metrics = self.reduce_metrics(metrics, self.distributed)
            meters.update(metrics)

            if epoch > self.meta_sta_epoch and (
                    (len(self.best_children_pool) < self.pool_size) or (prec1 > self.best_children_pool[-1][1] + 5) or
                    (prec1 > self.best_children_pool[-1][1] and cand_flops < self.best_children_pool[-1][2])):
                val_prec1 = prec1
                training_data = deepcopy(input_data[:self.slices].detach())
                if not self.best_children_pool:
                    features = deepcopy(output[:self.slices].detach())
                else:
                    features = deepcopy(teacher_output[:self.slices].detach())
                self.best_children_pool.append(
                    (val_prec1, prec1, cand_flops, training_data, F.softmax(features, dim=1)))
                self.arch_dict[(val_prec1, cand_flops)] = self.mutator._cache
                self.best_children_pool = sorted(self.best_children_pool, reverse=True)

            if len(self.best_children_pool) > self.pool_size:
                self.best_children_pool = sorted(self.best_children_pool, reverse=True)
                del self.best_children_pool[-1]

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.main_proc and self.log_frequency is not None and step % self.log_frequency == 0:
                logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

        if self.main_proc:
            for idx, i in enumerate(self.best_children_pool):
                logger.info("No.%s %s", idx, i[:4])

    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                self.mutator.reset()
                logits = self.model(x)
                loss = self.val_loss(logits, y)
                prec1, prec5 = self.accuracy(logits, y, topk=(1, 5))
                metrics = {"prec1": prec1, "prec5": prec5, "loss": loss}
                metrics = self.reduce_metrics(metrics, self.distributed)
                meters.update(metrics)

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.valid_loader), meters)
