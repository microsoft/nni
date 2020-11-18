# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

import time
import torchvision
import torch.nn.functional as F

from lib.utils.util import *


def train_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        prioritized_board,
        MetaMN,
        cfg,
        est=None,
        logger=None,
        lr_scheduler=None,
        saver=None,
        output_dir='',
        model_ema=None,
        local_rank=0):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    kd_losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        # get random architectures
        prob = prioritized_board.get_prob()
        random_cand = prioritized_board.get_cand_with_prob(prob)
        random_cand.insert(0, [0])
        random_cand.append([0])

        # evaluate FLOPs of candidates
        # cand_flops = est.get_flops(random_cand)

        # update meta matching networks
        # MetaMN.run_update(input, target, random_cand, model, optimizer,
        #                  prioritized_board, loss_fn, epoch, batch_idx)

        # get_best_teacher
        # meta_value, teacher_cand = prioritized_board.select_teacher(model, random_cand)

        if prioritized_board.board_size() == 0 or epoch <= cfg.SUPERNET.META_STA_EPOCH:
            output = model(input, random_cand)
            loss = loss_fn(output, target)
            kd_loss, teacher_output, teacher_cand = None, None, None
        else:
            output = model(input, random_cand)
            valid_loss = loss_fn(output, target)

            # get soft label from teacher cand
            with torch.no_grad():
                teacher_output = model(input, teacher_cand).detach()
                soft_label = F.softmax(teacher_output, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)

            loss = (meta_value * kd_loss + (2 - meta_value) * valid_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        if cfg.NUM_GPU == 1:
            reduced_loss = loss.data
        else:
            reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
            prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
            prec5 = reduce_tensor(prec5, cfg.NUM_GPU)

        # prioritized_board.update_prioritized_board(input, teacher_output, output, epoch, prec1, cand_flops, teacher_cand)

        torch.cuda.synchronize()

        if kd_loss is not None:
            kd_losses_m.update(kd_loss.item(), input.size(0))
        losses_m.update(reduced_loss.item(), input.size(0))
        prec1_m.update(prec1.item(), output.size(0))
        prec5_m.update(prec5.item(), output.size(0))
        batch_time_m.update(time.time() - end)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if last_batch or batch_idx % cfg.LOG_INTERVAL == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'KD-Loss: {kd_loss.val:>9.6f} ({kd_loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        kd_loss=kd_losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * cfg.NUM_GPU / batch_time_m.val,
                        rate_avg=input.size(0) * cfg.NUM_GPU / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if cfg.SAVE_IMAGES and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            batch_idx), padding=0, normalize=True)

        if saver is not None and cfg.RECOVERY_INTERVAL and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
            saver.save_recovery(model, optimizer, cfg, epoch,
                                model_ema=model_ema, batch_idx=batch_idx)

        end = time.time()

    if local_rank == 0:
        for idx, i in enumerate(prioritized_board.prioritized_board):
            logger.info("No.{} {}".format(idx, i[:4]))

    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        prioritized_board,
        cfg,
        log_suffix='',
        local_rank=0,
        logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    # get random child architecture
    random_cand = prioritized_board.get_cand_with_prob(None)
    random_cand.insert(0, [0])
    random_cand.append([0])

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input, random_cand)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = cfg.TTA
            if reduce_factor > 1:
                output = output.unfold(
                    0,
                    reduce_factor,
                    reduce_factor).mean(
                    dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            if cfg.NUM_GPU > 1:
                reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
                prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
                prec5 = reduce_tensor(prec5, cfg.NUM_GPU)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if local_rank == 0 and (
                    last_batch or batch_idx %
                    cfg.LOG_INTERVAL == 0):
                log_name = 'Test' + log_suffix
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx,
                        batch_time=batch_time_m, loss=losses_m,
                        top1=prec1_m, top5=prec5_m))

    metrics = OrderedDict(
        [('loss', losses_m.avg), ('prec1', prec1_m.avg), ('prec5', prec5_m.avg)])

    return metrics
