import time
import math
from datetime import timedelta
import torch
from torch import nn as nn
from nni.nas.pytorch.utils import AverageMeter

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
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


class Retrain:
    def __init__(self, model, optimizer, device, data_provider, n_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = data_provider.train
        self.valid_loader = data_provider.valid
        self.test_loader = data_provider.test
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        # train
        self.train()
        # validate
        self.validate(is_test=False)
        # test
        self.validate(is_test=True)

    def train_one_epoch(self, adjust_lr_func, train_log_func, label_smoothing=0.1):
        batch_time = AverageMeter('batch_time')
        data_time = AverageMeter('data_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        self.model.train()
        end = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            if label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(output, labels, label_smoothing)
            else:
                loss = self.criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.model.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 or i + 1 == len(self.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                print(batch_log)
        return top1, top5

    def train(self, validation_frequency=1):
        best_acc = 0
        nBatch = len(self.train_loader)

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
                batch_log = 'Train [{0}][{1}/{2}]\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                            'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                            'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                    format(epoch_ + 1, i, nBatch - 1,
                        batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
                batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                batch_log += '\tlr {lr:.5f}'.format(lr=lr)
                return batch_log
        
        def adjust_learning_rate(n_epochs, optimizer, epoch, batch=0, nBatch=None):
            """ adjust learning of a given optimizer and return the new learning rate """
            # cosine
            T_total = n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            # init_lr = 0.05
            new_lr = 0.5 * 0.05 * (1 + math.cos(math.pi * T_cur / T_total))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            return new_lr

        for epoch in range(self.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            end = time.time()
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: adjust_learning_rate(self.n_epochs, self.optimizer, epoch, i, nBatch),
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((self.n_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))
            
            if (epoch + 1) % validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(is_test=False)
                is_best = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'.\
                    format(epoch + 1, self.n_epochs, val_loss, val_acc, best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}'.\
                    format(val_acc5, top1=train_top1, top5=train_top5)
                print(val_log)
            else:
                is_best = False

    def validate(self, is_test=True):
        if is_test:
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader
        self.model.eval()
        batch_time = AverageMeter('batch_time')
        losses = AverageMeter('losses')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.model(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t'\
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'.\
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        return losses.avg, top1.avg, top5.avg