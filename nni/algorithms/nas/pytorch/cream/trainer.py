# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
import logging

from copy import deepcopy
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeterGroup

from .utils import accuracy, reduce_metrics

logger = logging.getLogger(__name__)


class CreamSupernetTrainer(Trainer):
    """
    This trainer trains a supernet and output prioritized architectures that can be used for other tasks.

    Parameters
    ----------
    model : nn.Module
        Model with mutables.
    loss : callable
        Called with logits and targets. Returns a loss tensor.
    val_loss : callable
        Called with logits and targets for validation only. Returns a loss tensor.
    optimizer : Optimizer
        Optimizer that optimizes the model.
    num_epochs : int
        Number of epochs of training.
    train_loader : iterablez
        Data loader of training. Raise ``StopIteration`` when one epoch is exhausted.
    valid_loader : iterablez
        Data loader of validation. Raise ``StopIteration`` when one epoch is exhausted.
    mutator : Mutator
        A mutator object that has been initialized with the model.
    batch_size : int
        Batch size.
    log_frequency : int
        Number of mini-batches to log metrics.
    meta_sta_epoch : int
        start epoch of using meta matching network to pick teacher architecture
    update_iter : int
        interval of updating meta matching networks
    slices : int
        batch size of mini training data in the process of training meta matching network
    pool_size : int
        board size
    pick_method : basestring
        how to pick teacher network
    choice_num : int
        number of operations in supernet
    sta_num : int
        layer number of each stage in supernet (5 stage in supernet)
    acc_gap : int
        maximum accuracy improvement to omit the limitation of flops
    flops_dict : Dict
        dictionary of each layer's operations in supernet
    flops_fixed : int
        flops of fixed part in supernet
    local_rank : int
        index of current rank
    callbacks : list of Callback
        Callbacks to plug into the trainer. See Callbacks.
    """

    def __init__(self, model, loss, val_loss,
                 optimizer, num_epochs, train_loader, valid_loader,
                 mutator=None, batch_size=64, log_frequency=None,
                 meta_sta_epoch=20, update_iter=200, slices=2,
                 pool_size=10, pick_method='meta', choice_num=6,
                 sta_num=(4, 4, 4, 4, 4), acc_gap=5,
                 flops_dict=None, flops_fixed=0, local_rank=0, callbacks=None):
        assert torch.cuda.is_available()
        super(CreamSupernetTrainer, self).__init__(model, mutator, loss, None,
                                                   optimizer, num_epochs, None, None,
                                                   batch_size, None, None, log_frequency, callbacks)
        self.model = model
        self.loss = loss
        self.val_loss = val_loss
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_frequency = log_frequency
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.num_epochs = num_epochs
        self.meta_sta_epoch = meta_sta_epoch
        self.update_iter = update_iter
        self.slices = slices
        self.pick_method = pick_method
        self.pool_size = pool_size
        self.local_rank = local_rank
        self.choice_num = choice_num
        self.sta_num = sta_num
        self.acc_gap = acc_gap
        self.flops_dict = flops_dict
        self.flops_fixed = flops_fixed

        self.current_student_arch = None
        self.current_teacher_arch = None
        self.main_proc = (local_rank == 0)
        self.current_epoch = 0

        self.prioritized_board = []

    # size of prioritized board
    def _board_size(self):
        return len(self.prioritized_board)

    # select teacher architecture according to the logit difference
    def _select_teacher(self):
        self._replace_mutator_cand(self.current_student_arch)

        if self.pick_method == 'top1':
            meta_value, teacher_cand = 0.5, sorted(
                self.prioritized_board, reverse=True)[0][3]
        elif self.pick_method == 'meta':
            meta_value, cand_idx, teacher_cand = -1000000000, -1, None
            for now_idx, item in enumerate(self.prioritized_board):
                inputx = item[4]
                output = torch.nn.functional.softmax(self.model(inputx), dim=1)
                weight = self.model.module.forward_meta(output - item[5])
                if weight > meta_value:
                    meta_value = weight
                    cand_idx = now_idx
                    teacher_cand = self.prioritized_board[cand_idx][3]
            assert teacher_cand is not None
            meta_value = torch.nn.functional.sigmoid(-weight)
        else:
            raise ValueError('Method Not supported')

        return meta_value, teacher_cand

    # check whether to update prioritized board
    def _isUpdateBoard(self, prec1, flops):
        if self.current_epoch <= self.meta_sta_epoch:
            return False

        if len(self.prioritized_board) < self.pool_size:
            return True

        if prec1 > self.prioritized_board[-1][1] + self.acc_gap:
            return True

        if prec1 > self.prioritized_board[-1][1] and flops < self.prioritized_board[-1][2]:
            return True

        return False

    # update prioritized board
    def _update_prioritized_board(self, inputs, teacher_output, outputs, prec1, flops):
        if self._isUpdateBoard(prec1, flops):
            val_prec1 = prec1
            training_data = deepcopy(inputs[:self.slices].detach())
            if len(self.prioritized_board) == 0:
                features = deepcopy(outputs[:self.slices].detach())
            else:
                features = deepcopy(
                    teacher_output[:self.slices].detach())
            self.prioritized_board.append(
                (val_prec1,
                 prec1,
                 flops,
                 self.current_student_arch,
                 training_data,
                 torch.nn.functional.softmax(
                     features,
                     dim=1)))
            self.prioritized_board = sorted(
                self.prioritized_board, reverse=True)

        if len(self.prioritized_board) > self.pool_size:
            del self.prioritized_board[-1]

    # only update student network weights
    def _update_student_weights_only(self, grad_1):
        for weight, grad_item in zip(
                self.model.module.rand_parameters(self.current_student_arch), grad_1):
            weight.grad = grad_item
        torch.nn.utils.clip_grad_norm_(
            self.model.module.rand_parameters(self.current_student_arch), 1)
        self.optimizer.step()
        for weight, grad_item in zip(
                self.model.module.rand_parameters(self.current_student_arch), grad_1):
            del weight.grad

    # only update meta networks weights
    def _update_meta_weights_only(self, teacher_cand, grad_teacher):
        for weight, grad_item in zip(self.model.module.rand_parameters(
                teacher_cand, self.pick_method == 'meta'), grad_teacher):
            weight.grad = grad_item

        # clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.module.rand_parameters(
                self.current_student_arch, self.pick_method == 'meta'), 1)

        self.optimizer.step()
        for weight, grad_item in zip(self.model.module.rand_parameters(
                teacher_cand, self.pick_method == 'meta'), grad_teacher):
            del weight.grad

    # simulate sgd updating
    def _simulate_sgd_update(self, w, g, optimizer):
        return g * optimizer.param_groups[-1]['lr'] + w

    # split training images into several slices
    def _get_minibatch_input(self, input):
        slice = self.slices
        x = deepcopy(input[:slice].clone().detach())
        return x

    # calculate 1st gradient of student architectures
    def _calculate_1st_gradient(self, kd_loss):
        self.optimizer.zero_grad()
        grad = torch.autograd.grad(
            kd_loss,
            self.model.module.rand_parameters(self.current_student_arch),
            create_graph=True)
        return grad

    # calculate 2nd gradient of meta networks
    def _calculate_2nd_gradient(self, validation_loss, teacher_cand, students_weight):
        self.optimizer.zero_grad()
        grad_student_val = torch.autograd.grad(
            validation_loss,
            self.model.module.rand_parameters(self.current_student_arch),
            retain_graph=True)

        grad_teacher = torch.autograd.grad(
            students_weight[0],
            self.model.module.rand_parameters(
                teacher_cand,
                self.pick_method == 'meta'),
            grad_outputs=grad_student_val)
        return grad_teacher

    # forward training data
    def _forward_training(self, x, meta_value):
        self._replace_mutator_cand(self.current_student_arch)
        output = self.model(x)

        with torch.no_grad():
            self._replace_mutator_cand(self.current_teacher_arch)
            teacher_output = self.model(x)
            soft_label = torch.nn.functional.softmax(teacher_output, dim=1)

        kd_loss = meta_value * \
            self._cross_entropy_loss_with_soft_target(output, soft_label)
        return kd_loss

    # calculate soft target loss
    def _cross_entropy_loss_with_soft_target(self, pred, soft_target):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

    # forward validation data
    def _forward_validation(self, input, target):
        slice = self.slices
        x = input[slice:slice * 2].clone()

        self._replace_mutator_cand(self.current_student_arch)
        output_2 = self.model(x)

        validation_loss = self.loss(output_2, target[slice:slice * 2])
        return validation_loss

    def _isUpdateMeta(self, batch_idx):
        isUpdate = True
        isUpdate &= (self.current_epoch > self.meta_sta_epoch)
        isUpdate &= (batch_idx > 0)
        isUpdate &= (batch_idx % self.update_iter == 0)
        isUpdate &= (self._board_size() > 0)
        return isUpdate

    def _replace_mutator_cand(self, cand):
        self.mutator._cache = cand

    # update meta matching networks
    def _run_update(self, input, target, batch_idx):
        if self._isUpdateMeta(batch_idx):
            x = self._get_minibatch_input(input)

            meta_value, teacher_cand = self._select_teacher()

            kd_loss = self._forward_training(x, meta_value)

            # calculate 1st gradient
            grad_1st = self._calculate_1st_gradient(kd_loss)

            # simulate updated student weights
            students_weight = [
                self._simulate_sgd_update(
                    p, grad_item, self.optimizer) for p, grad_item in zip(
                    self.model.module.rand_parameters(self.current_student_arch), grad_1st)]

            # update student weights
            self._update_student_weights_only(grad_1st)

            validation_loss = self._forward_validation(input, target)

            # calculate 2nd gradient
            grad_teacher = self._calculate_2nd_gradient(validation_loss, teacher_cand, students_weight)

            # update meta matching networks
            self._update_meta_weights_only(teacher_cand, grad_teacher)

            # delete internal variants
            del grad_teacher, grad_1st, x, validation_loss, kd_loss, students_weight

    def _get_cand_flops(self, cand):
        flops = 0
        for block_id, block in enumerate(cand):
            if block == 'LayerChoice1' or block_id == 'LayerChoice23':
                continue
            for idx, choice in enumerate(cand[block]):
                flops += self.flops_dict[block_id][idx] * (1 if choice else 0)
        return flops + self.flops_fixed

    def train_one_epoch(self, epoch):
        self.current_epoch = epoch
        meters = AverageMeterGroup()
        self.steps_per_epoch = len(self.train_loader)
        for step, (input_data, target) in enumerate(self.train_loader):
            self.mutator.reset()
            self.current_student_arch = self.mutator._cache

            input_data, target = input_data.cuda(), target.cuda()

            # calculate flops of current architecture
            cand_flops = self._get_cand_flops(self.mutator._cache)

            # update meta matching network
            self._run_update(input_data, target, step)

            if self._board_size() > 0:
                # select teacher architecture
                meta_value, teacher_cand = self._select_teacher()
                self.current_teacher_arch = teacher_cand

            # forward supernet
            if self._board_size() == 0 or epoch <= self.meta_sta_epoch:
                self._replace_mutator_cand(self.current_student_arch)
                output = self.model(input_data)

                loss = self.loss(output, target)
                kd_loss, teacher_output, teacher_cand = None, None, None
            else:
                self._replace_mutator_cand(self.current_student_arch)
                output = self.model(input_data)

                gt_loss = self.loss(output, target)

                with torch.no_grad():
                    self._replace_mutator_cand(self.current_teacher_arch)
                    teacher_output = self.model(input_data).detach()

                    soft_label = torch.nn.functional.softmax(teacher_output, dim=1)
                kd_loss = self._cross_entropy_loss_with_soft_target(output, soft_label)

                loss = (meta_value * kd_loss + (2 - meta_value) * gt_loss) / 2

            # update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update metrics
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            metrics = {"prec1": prec1, "prec5": prec5, "loss": loss}
            metrics = reduce_metrics(metrics)
            meters.update(metrics)

            # update prioritized board
            self._update_prioritized_board(input_data, teacher_output, output, metrics['prec1'], cand_flops)

            if self.main_proc and (step % self.log_frequency == 0 or step + 1 == self.steps_per_epoch):
                logger.info("Epoch [%d/%d] Step [%d/%d] %s", epoch + 1, self.num_epochs,
                            step + 1, len(self.train_loader), meters)

        if self.main_proc and self.num_epochs == epoch + 1:
            for idx, i in enumerate(self.prioritized_board):
                logger.info("No.%s %s", idx, i[:4])

    def validate_one_epoch(self, epoch):
        self.model.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.valid_loader):
                self.mutator.reset()
                logits = self.model(x)
                loss = self.val_loss(logits, y)
                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                metrics = {"prec1": prec1, "prec5": prec5, "loss": loss}
                metrics = reduce_metrics(metrics)
                meters.update(metrics)

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    logger.info("Epoch [%s/%s] Validation Step [%s/%s]  %s", epoch + 1,
                                self.num_epochs, step + 1, len(self.valid_loader), meters)
