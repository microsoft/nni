# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from loader import get_train_loaders, add_depth_channel
from models import UNetResNetV4, UNetResNetV5, UNetResNetV6
from lovasz_losses import lovasz_hinge
from focal_loss import FocalLoss2d
from postprocessing import binarize, crop_image, resize_image
from metrics import intersection_over_union, intersection_over_union_thresholds
import settings

MODEL_DIR = settings.MODEL_DIR
focal_loss2d = FocalLoss2d()

def weighted_loss(args, output, target, epoch=0):
    mask_output, salt_output = output
    mask_target, salt_target = target

    lovasz_loss = lovasz_hinge(mask_output, mask_target)
    focal_loss = focal_loss2d(mask_output, mask_target)

    focal_weight = 0.2

    if salt_output is not None and args.train_cls:
        salt_loss = F.binary_cross_entropy_with_logits(salt_output, salt_target)
        return salt_loss, focal_loss.item(), lovasz_loss.item(), salt_loss.item(), lovasz_loss.item() + focal_loss.item()*focal_weight

    return lovasz_loss+focal_loss*focal_weight, focal_loss.item(), lovasz_loss.item(), 0., lovasz_loss.item() + focal_loss.item()*focal_weight

def train(args):
    print('start training...')

    """@nni.variable(nni.choice('UNetResNetV4', 'UNetResNetV5', 'UNetResNetV6'), name=model_name)"""
    model_name = args.model_name

    model = eval(model_name)(args.layers, num_filters=args.nf)
    model_subdir = args.pad_mode
    if args.meta_version == 2:
        model_subdir = args.pad_mode+'_meta2'
    if args.exp_name is None:
        model_file = os.path.join(MODEL_DIR, model.name,model_subdir, 'best_{}.pth'.format(args.ifold))
    else:
        model_file = os.path.join(MODEL_DIR, args.exp_name, model.name, model_subdir, 'best_{}.pth'.format(args.ifold))

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.init_ckp is not None:
        CKP = args.init_ckp
    else:
        CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=args.batch_size, dev_mode=args.dev_mode, \
        pad_mode=args.pad_mode, meta_version=args.meta_version, pseudo_label=args.pseudo, depths=args.depths)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)

    print('epoch |   lr    |   %       |  loss  |  avg   | f loss | lovaz  |  iou   | iout   |  best  | time | save |  salt  |')

    best_iout, _iou, _f, _l, _salt, best_mix_score = validate(args, model, val_loader, args.start_epoch)
    print('val   |         |           |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      | {:.4f} |'.format(
        _f, _l, _iou, best_iout, best_iout, _salt))
    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_iout)
    else:
        lr_scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            img, target, salt_target = data
            if args.depths:
                add_depth_channel(img, args.pad_mode)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            optimizer.zero_grad()
            output, salt_out = model(img)

            loss, *_ = weighted_loss(args, (output, salt_out), (target, salt_target), epoch=epoch)
            loss.backward()

            if args.optim == 'Adam' and args.adamw:
                wd = 0.0001
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data = param.data.add(-wd * group['lr'], param.data)

            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

        iout, iou, focal_loss, lovaz_loss, salt_loss, mix_score = validate(args, model, val_loader, epoch=epoch)
        """@nni.report_intermediate_result(iout)"""

        _save_ckp = ''
        if iout > best_iout:
            best_iout = iout
            torch.save(model.state_dict(), model_file)
            _save_ckp = '*'
        if args.store_loss_model and mix_score > best_mix_score:
            best_mix_score = mix_score
            torch.save(model.state_dict(), model_file+'_loss')
            _save_ckp += '.'
        print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
            focal_loss, lovaz_loss, iou, iout, best_iout, (time.time() - bg) / 60, _save_ckp, salt_loss))

        model.train()

        if args.lrs == 'plateau':
            lr_scheduler.step(best_iout)
        else:
            lr_scheduler.step()

    del model, train_loader, val_loader, optimizer, lr_scheduler
    """@nni.report_final_result(best_iout)"""

def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader, epoch=0, threshold=0.5):
    model.eval()
    outputs = []
    focal_loss, lovaz_loss, salt_loss, w_loss = 0, 0, 0, 0
    with torch.no_grad():
        for img, target, salt_target in val_loader:
            if args.depths:
                add_depth_channel(img, args.pad_mode)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            output, salt_out = model(img)

            _, floss, lovaz, _salt_loss, _w_loss = weighted_loss(args, (output, salt_out), (target, salt_target), epoch=epoch)
            focal_loss += floss
            lovaz_loss += lovaz
            salt_loss += _salt_loss
            w_loss += _w_loss
            output = torch.sigmoid(output)

            for o in output.cpu():
                outputs.append(o.squeeze().numpy())

    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    # y_pred, list of np array, each np array's shape is 101,101
    y_pred = generate_preds(args, outputs, (settings.ORIG_H, settings.ORIG_W), threshold)

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)

    return iout_score, iou_score, focal_loss / n_batches, lovaz_loss / n_batches, salt_loss / n_batches, iout_score*4 - w_loss


def generate_preds(args, outputs, target_size, threshold=0.5):
    preds = []

    for output in outputs:
        if args.pad_mode == 'resize':
            cropped = resize_image(output, target_size=target_size)
        else:
            cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TGS Salt segmentation')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--ifolds', default='0', type=str, help='kfold indices')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--pad_mode', default='edge', choices=['reflect', 'edge', 'resize'], help='pad method')
    parser.add_argument('--exp_name', default=None, type=str, help='exp name')
    parser.add_argument('--model_name', default='UNetResNetV4', type=str, help='')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--store_loss_model', action='store_true')
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--meta_version', default=2, type=int, help='meta version')
    parser.add_argument('--pseudo', action='store_true')
    parser.add_argument('--depths', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--adamw', action='store_true')

    args = parser.parse_args()

    '''@nni.get_next_parameter()'''

    print(args)
    ifolds = [int(x) for x in args.ifolds.split(',')]
    print(ifolds)

    for i in ifolds:
        args.ifold = i
        train(args)
