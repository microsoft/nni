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

import numpy as np
from pycocotools import mask as cocomask
from utils import get_segmentations


def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    ious = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / len(iou)
        ious.append(iou_mean)
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    iouts = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)
