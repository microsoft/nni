# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import numpy as np


def accuracy(preds, target):
    """
    Calculate the NME (Normalized Mean Error).

    Parameters
    ----------
    preds : numpy array
        the predicted landmarks
    target : numpy array
        the ground truth of landmarks

    Returns
    -------
    output: float32
        the nme value
    output: list
        the list of l2 distances
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N).astype(np.float32)

    for i in range(N):
        pts_pred, pts_gt = (
            preds[i],
            target[i],
        )
        if L == 19:
            # aflw
            interocular = 34
        elif L == 29:
            # cofw
            interocular = np.linalg.norm(pts_gt[8] - pts_gt[9])
        elif L == 68:
            # interocular
            interocular = np.linalg.norm(pts_gt[36] - pts_gt[45])
        elif L == 98:
            # euclidean dis from left eye to right eye
            interocular = np.linalg.norm(pts_gt[60] - pts_gt[72])
        elif L == 106:
            # euclidean dis from left eye to right eye
            interocular = np.linalg.norm(pts_gt[35] - pts_gt[93])
        else:
            raise ValueError("Number of landmarks is wrong")
        
        pred_dis = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1))
        rmse[i] = pred_dis / (interocular * L)

    return np.mean(rmse), rmse
