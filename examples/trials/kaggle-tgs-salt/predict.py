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
import glob
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_test_loader, add_depth_channel
from models import UNetResNetV4, UNetResNetV5, UNetResNetV6, UNet7, UNet8
from postprocessing import crop_image, binarize, resize_image
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import create_submission

def do_tta_predict(args, model, ckp_path, tta_num=4):
    '''
    return 18000x128x128 np array
    '''
    model.eval()
    preds = []
    meta = None

    # i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
    for flip_index in range(tta_num):
        print('flip_index:', flip_index)
        test_loader = get_test_loader(args.batch_size, index=flip_index, dev_mode=False, pad_mode=args.pad_mode)
        meta = test_loader.meta
        outputs = None
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                add_depth_channel(img, args.pad_mode)
                img = img.cuda()
                output, _ = model(img)
                output = torch.sigmoid(output)
                if outputs is None:
                    outputs = output.squeeze()
                else:
                    outputs = torch.cat([outputs, output.squeeze()], 0)

                print('{} / {}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
        outputs = outputs.cpu().numpy()
        # flip back masks
        if flip_index == 1:
            outputs = np.flip(outputs, 2)
        elif flip_index == 2:
            outputs = np.flip(outputs, 1)
        elif flip_index == 3:
            outputs = np.flip(outputs, 2)
            outputs = np.flip(outputs, 1)
        #print(outputs.shape)
        preds.append(outputs)

    parent_dir = ckp_path+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred.npy')

    model_pred_result = np.mean(preds, 0)
    np.save(np_file, model_pred_result)

    return model_pred_result, meta

def predict(args, model, checkpoint, out_file):
    print('predicting {}...'.format(checkpoint))
    pred, meta = do_tta_predict(args, model, checkpoint, tta_num=2)
    print(pred.shape)
    y_pred_test = generate_preds(pred, (settings.ORIG_H, settings.ORIG_W), pad_mode=args.pad_mode)

    submission = create_submission(meta, y_pred_test)
    submission.to_csv(out_file, index=None, encoding='utf-8')


def ensemble(args, model, checkpoints):
    preds = []
    meta = None
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        model = model.cuda()
        print('predicting...', checkpoint)

        pred, meta = do_tta_predict(args, model, checkpoint, tta_num=2)
        preds.append(pred)

    y_pred_test = generate_preds(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W), args.pad_mode)

    submission = create_submission(meta, y_pred_test)
    submission.to_csv(args.sub_file, index=None, encoding='utf-8')

def ensemble_np(args, np_files, save_np=None):
    preds = []
    for np_file in np_files:
        pred = np.load(np_file)
        print(np_file, pred.shape)
        preds.append(pred)

    y_pred_test = generate_preds(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W), args.pad_mode)

    if save_np is not None:
        np.save(save_np, np.mean(preds, 0))

    meta = get_test_loader(args.batch_size, index=0, dev_mode=False, pad_mode=args.pad_mode).meta

    submission = create_submission(meta, y_pred_test)
    submission.to_csv(args.sub_file, index=None, encoding='utf-8')

def generate_preds(outputs, target_size, pad_mode, threshold=0.5):
    preds = []

    for output in outputs:
        #print(output.shape)
        if pad_mode == 'resize':
            cropped = resize_image(output, target_size=target_size)
        else:
            cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds


def ensemble_predict(args):
    model = eval(args.model_name)(args.layers, num_filters=args.nf)

    checkpoints = [
        r'D:\data\salt\models\pseudo\UNetResNetV4_34\edge\best_5.pth',
        r'D:\data\salt\models\pseudo\UNetResNetV4_34\edge\best_6.pth',
        r'D:\data\salt\models\pseudo\UNetResNetV4_34\edge\best_8.pth',
        r'D:\data\salt\models\pseudo\UNetResNetV4_34\edge\best_9.pth'
    ]
    print(checkpoints)

    ensemble(args, model, checkpoints)

def ensemble_np_results(args):
    np_files1 = glob.glob(r'D:\data\salt\models\depths\UNetResNetV5_50\edge\*pth_out\*.npy')
    np_files2 = glob.glob(r'D:\data\salt\models\depths\UNetResNetV4_34\edge\*pth_out\*.npy')
    np_files3 = glob.glob(r'D:\data\salt\models\depths\UNetResNetV6_34\edge\*pth_out\*.npy')
    np_files6 = glob.glob(r'D:\data\salt\models\ensemble\*.npy')
    np_files = np_files1 + np_files2 + np_files3 + np_files6
    print(np_files)
    ensemble_np(args, np_files)

def predict_model(args):
    model = eval(args.model_name)(args.layers, num_filters=args.nf)
    model_subdir = args.pad_mode
    if args.meta_version == 2:
        model_subdir = args.pad_mode+'_meta2'
    if args.exp_name is None:
        model_file = os.path.join(settings.MODEL_DIR, model.name,model_subdir, 'best_{}.pth'.format(args.ifold))
    else:
        model_file = os.path.join(settings.MODEL_DIR, args.exp_name, model.name, model_subdir, 'best_{}.pth'.format(args.ifold))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    else:
        raise ValueError('model file not found: {}'.format(model_file))
    model = model.cuda()
    predict(args, model, model_file, args.sub_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--model_name', required=True, type=str, help='')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--nf', default=32, type=int, help='num_filters param for model')
    parser.add_argument('--ifold', required=True, type=int, help='kfold indices')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--pad_mode', required=True, choices=['reflect', 'edge', 'resize'], help='pad method')
    parser.add_argument('--exp_name', default='depths', type=str, help='exp name')
    parser.add_argument('--meta_version', default=2, type=int, help='meta version')
    parser.add_argument('--sub_file', default='all_ensemble.csv', type=str, help='submission file')

    args = parser.parse_args()

    predict_model(args)
    #ensemble_predict(args)
    #ensemble_np_results(args)
