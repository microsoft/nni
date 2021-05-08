# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch

import numpy as np
import torch.nn as nn


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


class PFLDLoss(nn.Module):
    """Weighted loss of L2 distance with the pose angle for PFLD."""

    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, euler_angle_gt, angle, landmarks):
        """
        Calculate weighted L2 loss for PFLD.

        Parameters
        ----------
        landmark_gt : tensor
            the ground truth of landmarks
        euler_angle_gt : tensor
            the ground truth of pose angle
        angle : tensor
            the predicted pose angle
        landmarks : float32
            the predicted landmarks

        Returns
        -------
        output: tensor
            the weighted L2 loss
        output: tensor
            the normal L2 loss
        """
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        l2_distant = torch.sum((landmark_gt - landmarks) ** 2, axis=1)

        return torch.mean(weight_angle * l2_distant), torch.mean(l2_distant)


def bounded_regress_loss(
    landmark_gt, landmarks_t, landmarks_s, reg_m=0.5, br_alpha=0.05
):
    """
    Calculate the Bounded Regression Loss for Knowledge Distillation.

    Parameters
    ----------
    landmark_gt : tensor
        the ground truth of landmarks
    landmarks_t : tensor
        the predicted landmarks of teacher
    landmarks_s : tensor
        the predicted landmarks of student
    reg_m : float32
        the value to control the regresion constraint
    br_alpha : float32
        the balance value for kd loss

    Returns
    -------
    output: tensor
        the bounded regression loss
    """
    l2_dis_s = (landmark_gt - landmarks_s).pow(2).sum(1)
    l2_dis_s_m = l2_dis_s + reg_m

    l2_dis_t = (landmark_gt - landmarks_t).pow(2).sum(1)
    br_loss = l2_dis_s[l2_dis_s_m > l2_dis_t].sum()

    return br_loss * br_alpha
