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

import math
import time

import torch
from torch import nn as nn

from nni.nas.pytorch.base_trainer import BaseTrainer
from nni.nas.utils import AverageMeter
from .mutator import ProxylessNasMutator


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    """
    Parameters
    ----------
    pred :
    target :
    label_smoothing :

    Returns
    -------
    """
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k

    Parameters
    ----------
    output :
    target :
    topk :

    Returns
    -------
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ProxylessNasTrainer(BaseTrainer):
    def __init__(self, model, model_optim, train_loader, valid_loader, device,
                 n_epochs=150, init_lr=0.05, arch_init_type='normal', arch_init_ratio=1e-3,
                 arch_optim_lr=1e-3, arch_weight_decay=0, warmup=True, warmup_epochs=25,
                 arch_valid_frequency=1):
        """
        Parameters
        ----------
        model : pytorch model
        model_optim : pytorch optimizer
        train_loader : pytorch data loader
        valid_loader : pytorch data loader
        device : device
        n_epochs : int
        init_lr : float
            init learning rate for training the model
        arch_init_type : str
            the way to init architecture parameters
        arch_init_ratio : float
            the ratio to init architecture parameters
        arch_optim_lr : float
            learning rate of the architecture parameters optimizer
        arch_weight_decay : float
            weight decay of the architecture parameters optimizer
        warmup : bool
            whether to do warmup
        warmup_epochs : int
            the number of epochs to do in warmup
        arch_valid_frequency : int
            frequency of printing validation result
        """
        self.model = model
        self.model_optim = model_optim
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.warmup = warmup
        self.warmup_epochs = warmup_epochs
        self.arch_valid_frequency = arch_valid_frequency

        self.train_epochs = 120
        self.lr_max = 0.05
        self.label_smoothing = 0.1
        self.valid_batch_size = 500
        self.arch_grad_valid_batch_size = 2 # 256
        # update architecture parameters every this number of minibatches
        self.grad_update_arch_param_every = 5
        # the number of steps per architecture parameter update
        self.grad_update_steps = 1

        # init mutator
        self.mutator = ProxylessNasMutator(model)
        self._valid_iter = None

        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # TODO: arch search configs

        self._init_arch_params(arch_init_type, arch_init_ratio)

        # build architecture optimizer
        self.arch_optimizer = torch.optim.Adam(self.mutator.get_architecture_parameters(),
                                               arch_optim_lr,
                                               weight_decay=arch_weight_decay)

        self.criterion = nn.CrossEntropyLoss()

    def _init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.mutator.get_architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def _validate(self):
        self.valid_loader.batch_sampler.batch_size = self.valid_batch_size
        self.valid_loader.batch_sampler.drop_last = False

        self.mutator.set_chosen_op_active()
        # test on validation set under train mode
        self.model.train()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
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
                    test_log = 'Valid' + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(self.valid_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    # return top5:
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        return losses.avg, top1.avg, top5.avg

    def _warm_up(self):
        data_loader = self.train_loader
        nBatch = len(data_loader)
        T_total = self.warmup_epochs * nBatch # total num of batches

        for epoch in range(self.warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter('batch_time')
            data_time = AverageMeter('data_time')
            losses = AverageMeter('losses')
            top1 = AverageMeter('top1')
            top5 = AverageMeter('top5')
            # switch to train mode
            self.model.train()

            end = time.time()
            print('=====================_warm_up, epoch: ', epoch)
            for i, (images, labels) in enumerate(data_loader):
                #print('=====================_warm_up, minibatch i: ', i)
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * self.lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.model_optim.param_groups:
                    param_group['lr'] = warmup_lr
                images, labels = images.to(self.device), labels.to(self.device)
                print(images, labels)
                # compute output
                self.mutator.reset_binary_gates() # random sample binary gates
                output = self.model(images)
                if self.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(output, labels, self.label_smoothing)
                else:
                    loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.model.zero_grad()
                loss.backward()
                self.model_optim.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    print(batch_log)
            val_loss, val_top1, val_top5 = self._validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}M'. \
                format(epoch + 1, self.warmup_epochs, val_loss, val_top1, val_top5, top1=top1, top5=top5)
            print(val_log)

    def _get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.grad_update_arch_param_every == 0:
                schedule[i] = self.grad_update_steps
        return schedule

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        T_total = self.n_epochs * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        return lr

    def _adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """
        Adjust learning of a given optimizer and return the new learning rate
        """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def _train(self):
        nBatch = len(self.train_loader)
        arch_param_num = self.mutator.num_arch_params()
        binary_gates_num = self.mutator.num_arch_params()
        #weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: xx' %
            (arch_param_num, binary_gates_num)
        )

        update_schedule = self._get_update_schedule(nBatch)

        for epoch in range(self.train_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter('batch_time')
            data_time = AverageMeter('data_time')
            losses = AverageMeter('losses')
            top1 = AverageMeter('top1')
            top5 = AverageMeter('top5')
            entropy = AverageMeter('entropy')
            # switch to train mode
            self.model.train()

            end = time.time()
            for i, (images, labels) in enumerate(self.train_loader):
                data_time.update(time.time() - end)
                lr = self._adjust_learning_rate(self.model_optim, epoch, batch=i, nBatch=nBatch)
                # network entropy
                #net_entropy = self.mutator.entropy()
                #entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters
                images, labels = images.to(self.device), labels.to(self.device)
                self.mutator.reset_binary_gates()
                output = self.model(images)
                if self.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(output, labels, self.label_smoothing)
                else:
                    loss = self.criterion(output, labels)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                self.model.zero_grad()
                loss.backward()
                self.model_optim.step()
                # TODO: if epoch > 0:
                if epoch >= 0:
                    for _ in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        # GradientArchSearchConfig
                        arch_loss, exp_value = self._gradient_step()
                        used_time = time.time() - start_time
                        log_str = 'Architecture [%d-%d]\t Time %.4f\t Loss %.4f\t null %s' % \
                                    (epoch + 1, i, used_time, arch_loss, exp_value)
                        print(log_str)
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % 10 == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr)
                    print(batch_log)
            # TODO: print current network architecture
            # validate
            if (epoch + 1) % self.arch_valid_frequency == 0:
                val_loss, val_top1, val_top5 = self._validate()
                val_log = 'Valid [{0}]\tloss {1:.3f}\ttop-1 acc {2:.3f} \ttop-5 acc {3:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}M'. \
                    format(epoch + 1, val_loss, val_top1,
                           val_top5, entropy=entropy, top1=top1, top5=top5)
                print(val_log)
        # convert to normal network according to architecture parameters

    def _valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    def _gradient_step(self):
        self.valid_loader.batch_sampler.batch_size = self.arch_grad_valid_batch_size
        self.valid_loader.batch_sampler.drop_last = True
        self.model.train()
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self._valid_next_batch()
        images, labels = images.to(self.device), labels.to(self.device)
        time2 = time.time()  # time
        self.mutator.reset_binary_gates()
        output = self.model(images)
        time3 = time.time()
        ce_loss = self.criterion(output, labels)
        expected_value = None
        loss = ce_loss
        self.model.zero_grad()
        loss.backward()
        self.mutator.set_arch_param_grad()
        self.arch_optimizer.step()
        time4 = time.time()
        print('(%.4f, %.4f, %.4f)' % (time2 - time1, time3 - time2, time4 - time3))
        return loss.data.item(), expected_value.item() if expected_value is not None else None

    def train(self):
        if self.warmup:
            self._warm_up()
        self._train()

    def export(self):
        pass

    def validate(self):
        raise NotImplementedError

    def train_and_validate(self):
        raise NotImplementedError
