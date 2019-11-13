# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import copy
import math

import torch
from torch import nn as nn

from nni.nas.pytorch.trainer import Trainer
from nni.nas.utils import AverageMeterGroup, auto_device
from .mutator import ProxylessNasMutator


class ProxylessNasTrainer(Trainer):
    def __init__(self, model, model_optim, train_loader, device):
        self.model = model
        self.model_optim = model_optim
        self.train_loader = train_loader
        self.device = device

        # TODO: arch search configs

        self._init_arch_params()

        # build architecture optimizer
        self.arch_optimizer = torch.optim.Adam(self._architecture_parameters(), 1e-3, weight_decay=0)

        self.warmup = True
        self.warmup_epoch = 0

    def _architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def _init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self._architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def _warm_up(self, warmup_epochs=25):
        lr_max = 0.05
        data_loader = self.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch # total num of batches

        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.model.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.model_optim.param_groups:
                    param_group['lr'] = warmup_lr
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                self._reset_binary_gates() # random sample binary gates
                # TODO: 
                #self._unused_modules_off() # remove unused module for speedup
                output = self.model(images)

    def _reset_binary_gates(self):
        for m in self.

    def train(self):
        pass

    def export(self):
        pass
