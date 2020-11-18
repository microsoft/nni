# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import torch
import torch.nn.functional as F

from copy import deepcopy

from lib.utils.util import cross_entropy_loss_with_soft_target


class MetaMatchingNetwork():
    def __init__(self, cfg):
        self.cfg = cfg

    # only update student network weights
    def update_student_weights_only(
            self,
            random_cand,
            grad_1,
            optimizer,
            model):
        for weight, grad_item in zip(
                model.module.rand_parameters(random_cand), grad_1):
            weight.grad = grad_item
        torch.nn.utils.clip_grad_norm_(
            model.module.rand_parameters(random_cand), 1)
        optimizer.step()
        for weight, grad_item in zip(
                model.module.rand_parameters(random_cand), grad_1):
            del weight.grad

    # only update meta networks weights
    def update_meta_weights_only(
            self,
            random_cand,
            teacher_cand,
            model,
            optimizer,
            grad_teacher):
        for weight, grad_item in zip(model.module.rand_parameters(
                teacher_cand, self.cfg.SUPERNET.PICK_METHOD == 'meta'), grad_teacher):
            weight.grad = grad_item

        # clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.module.rand_parameters(
                random_cand, self.cfg.SUPERNET.PICK_METHOD == 'meta'), 1)

        optimizer.step()
        for weight, grad_item in zip(model.module.rand_parameters(
                teacher_cand, self.cfg.SUPERNET.PICK_METHOD == 'meta'), grad_teacher):
            del weight.grad

    # simulate sgd updating
    def simulate_sgd_update(self, w, g, optimizer):
        return g * optimizer.param_groups[-1]['lr'] + w

    # split training images into several slices
    def get_minibatch_input(self, input):
        slice = self.cfg.SUPERNET.SLICE
        x = deepcopy(input[:slice].clone().detach())
        return x

    def calculate_1st_gradient(self, kd_loss, model, random_cand, optimizer):
        optimizer.zero_grad()
        grad = torch.autograd.grad(
            kd_loss,
            model.module.rand_parameters(random_cand),
            create_graph=True)
        return grad

    def calculate_2nd_gradient(
            self,
            validation_loss,
            model,
            optimizer,
            random_cand,
            teacher_cand,
            students_weight):
        optimizer.zero_grad()
        grad_student_val = torch.autograd.grad(
            validation_loss,
            model.module.rand_parameters(random_cand),
            retain_graph=True)

        grad_teacher = torch.autograd.grad(
            students_weight[0],
            model.module.rand_parameters(
                teacher_cand,
                self.cfg.SUPERNET.PICK_METHOD == 'meta'),
            grad_outputs=grad_student_val)
        return grad_teacher

    # forward training data
    def forward_training(
            self,
            x,
            model,
            random_cand,
            teacher_cand,
            meta_value):
        output = model(x, random_cand)
        with torch.no_grad():
            teacher_output = model(x, teacher_cand)
            soft_label = F.softmax(teacher_output, dim=1)
        kd_loss = meta_value * \
            cross_entropy_loss_with_soft_target(output, soft_label)
        return kd_loss

    # forward validation data
    def forward_validation(self, input, target, random_cand, model, loss_fn):
        slice = self.cfg.SUPERNET.SLICE
        x = input[slice:slice * 2].clone()
        output_2 = model(x, random_cand)
        validation_loss = loss_fn(output_2, target[slice:slice * 2])
        return validation_loss

    def isUpdate(self, current_epoch, batch_idx, prioritized_board):
        isUpdate = True
        isUpdate &= (current_epoch > self.cfg.SUPERNET.META_STA_EPOCH)
        isUpdate &= (batch_idx > 0)
        isUpdate &= (batch_idx % self.cfg.SUPERNET.UPDATE_ITER == 0)
        isUpdate &= (prioritized_board.board_size() > 0)
        return isUpdate

    # update meta matching networks
    def run_update(self, input, target, random_cand, model, optimizer,
                   prioritized_board, loss_fn, current_epoch, batch_idx):
        if self.isUpdate(current_epoch, batch_idx, prioritized_board):
            x = self.get_minibatch_input(input)

            meta_value, teacher_cand = prioritized_board.select_teacher(
                model, random_cand)

            kd_loss = self.forward_training(
                x, model, random_cand, teacher_cand, meta_value)

            # calculate 1st gradient
            grad_1st = self.calculate_1st_gradient(
                kd_loss, model, random_cand, optimizer)

            # simulate updated student weights
            students_weight = [
                self.simulate_sgd_update(
                    p, grad_item, optimizer) for p, grad_item in zip(
                    model.module.rand_parameters(random_cand), grad_1st)]

            # update student weights
            self.update_student_weights_only(
                random_cand, grad_1st, optimizer, model)

            validation_loss = self.forward_validation(
                input, target, random_cand, model, loss_fn)

            # calculate 2nd gradient
            grad_teacher = self.calculate_2nd_gradient(
                validation_loss, model, optimizer, random_cand, teacher_cand, students_weight)

            # update meta matching networks
            self.update_meta_weights_only(
                random_cand, teacher_cand, model, optimizer, grad_teacher)

            # delete internal variants
            del grad_teacher, grad_1st, x, validation_loss, kd_loss, students_weight
