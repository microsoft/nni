# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import numpy as np
import torch.nn.functional as F

from copy import deepcopy


class PrioritizedBoard():
    def __init__(self, cfg, CHOICE_NUM=6, sta_num=(4, 4, 4, 4, 4), acc_gap=5):
        self.cfg = cfg
        self.prioritized_board = []
        self.choice_num = CHOICE_NUM
        self.sta_num = sta_num
        self.acc_gap = acc_gap

    # select teacher from prioritized board

    def select_teacher(self, model, random_cand):
        if self.cfg.SUPERNET.PICK_METHOD == 'top1':
            meta_value, teacher_cand = 0.5, sorted(
                self.prioritized_board, reverse=True)[0][3]
        elif self.cfg.SUPERNET.PICK_METHOD == 'meta':
            meta_value, cand_idx, teacher_cand = -1000000000, -1, None
            for now_idx, item in enumerate(self.prioritized_board):
                inputx = item[4]
                output = F.softmax(model(inputx, random_cand), dim=1)
                weight = model.module.forward_meta(output - item[5])
                if weight > meta_value:
                    meta_value = weight
                    cand_idx = now_idx
                    teacher_cand = self.prioritized_board[cand_idx][3]
            assert teacher_cand is not None
            meta_value = F.sigmoid(-weight)
        else:
            raise ValueError('Method Not supported')

        return meta_value, teacher_cand

    def board_size(self):
        return len(self.prioritized_board)

    # get prob from config file

    def get_prob(self):
        if self.cfg.SUPERNET.HOW_TO_PROB == 'even' or (
            self.cfg.SUPERNET.HOW_TO_PROB == 'teacher' and len(
                self.prioritized_board) == 0):
            return None
        elif self.cfg.SUPERNET.HOW_TO_PROB == 'pre_prob':
            return self.cfg.SUPERNET.PRE_PROB
        elif self.cfg.SUPERNET.HOW_TO_PROB == 'teacher':
            op_dict = {}
            for i in range(self.choice_num):
                op_dict[i] = 0
            for item in self.prioritized_board:
                cand = item[3]
                for block in cand:
                    for op in block:
                        op_dict[op] += 1
            sum_op = 0
            for i in range(self.choice_num):
                sum_op = sum_op + op_dict[i]
            prob = []
            for i in range(self.choice_num):
                prob.append(float(op_dict[i]) / sum_op)
            del op_dict, sum_op
            return prob

    # sample random architecture

    def get_cand_with_prob(self, prob=None):
        if prob is None:
            get_random_cand = [
                np.random.choice(
                    self.choice_num,
                    item).tolist() for item in self.sta_num]
        else:
            get_random_cand = [
                np.random.choice(
                    self.choice_num,
                    item,
                    prob).tolist() for item in self.sta_num]

        return get_random_cand

    def isUpdate(self, current_epoch, prec1, flops):
        if current_epoch <= self.cfg.SUPERNET.META_STA_EPOCH:
            return False

        if len(self.prioritized_board) < self.cfg.SUPERNET.POOL_SIZE:
            return True

        if prec1 > self.prioritized_board[-1][1] + self.acc_gap:
            return True

        if prec1 > self.prioritized_board[-1][1] and flops < self.prioritized_board[-1][2]:
            return True

        return False

    def update_prioritized_board(
            self,
            inputs,
            teacher_output,
            outputs,
            current_epoch,
            prec1,
            flops,
            cand):
        if self.isUpdate(current_epoch, prec1, flops):
            val_prec1 = prec1
            training_data = deepcopy(inputs[:self.cfg.SUPERNET.SLICE].detach())
            if len(self.prioritized_board) == 0:
                features = deepcopy(outputs[:self.cfg.SUPERNET.SLICE].detach())
            else:
                features = deepcopy(
                    teacher_output[:self.cfg.SUPERNET.SLICE].detach())
            self.prioritized_board.append(
                (val_prec1,
                 prec1,
                 flops,
                 cand,
                 training_data,
                 F.softmax(
                     features,
                     dim=1)))
            self.prioritized_board = sorted(
                self.prioritized_board, reverse=True)

        if len(self.prioritized_board) > self.cfg.SUPERNET.POOL_SIZE:
            self.prioritized_board = sorted(
                self.prioritized_board, reverse=True)
            del self.prioritized_board[-1]
