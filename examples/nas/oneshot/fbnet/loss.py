# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch
from torch import nn


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
