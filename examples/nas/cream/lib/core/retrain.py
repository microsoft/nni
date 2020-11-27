import os
import time
import torch
import torchvision

from collections import OrderedDict

from lib.utils.util import AverageMeter, accuracy, reduce_tensor

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, cfg,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False,
        model_ema=None, logger=None, writer=None, local_rank=0):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    optimizer.zero_grad()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = loss_fn(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        if cfg.NUM_GPU > 1:
            reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
            prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
            prec5 = reduce_tensor(prec5, cfg.NUM_GPU)
        else:
            reduced_loss = loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        prec1_m.update(prec1.item(), output.size(0))
        prec5_m.update(prec5.item(), output.size(0))

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg.LOG_INTERVAL == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{}] '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f}) '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                    'LR: {lr:.3e}'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        loss=losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) *
                        cfg.NUM_GPU /
                        batch_time_m.val,
                        rate_avg=input.size(0) *
                        cfg.NUM_GPU /
                        batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                writer.add_scalar(
                    'Loss/train',
                    prec1_m.avg,
                    epoch *
                    len(loader) +
                    batch_idx)
                writer.add_scalar(
                    'Accuracy/train',
                    prec1_m.avg,
                    epoch *
                    len(loader) +
                    batch_idx)
                writer.add_scalar(
                    'Learning_Rate',
                    optimizer.param_groups[0]['lr'],
                    epoch * len(loader) + batch_idx)

                if cfg.SAVE_IMAGES and output_dir:
                    torchvision.utils.save_image(
                        input, os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            batch_idx), padding=0, normalize=True)

        if saver is not None and cfg.RECOVERY_INTERVAL and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
            saver.save_recovery(
                model,
                optimizer,
                cfg,
                epoch,
                model_ema=model_ema,
                use_amp=use_amp,
                batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates,
                metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])
