# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import torch
from torch import nn


class PFLDLoss(nn.Module):
    """
    Weighted loss of L2 distance with the pose angle for PFLD. 
    """

    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, euler_angle_gt, angle, landmarks):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        l2_distant = torch.sum((landmark_gt - landmarks) ** 2, axis=1)
        return torch.mean(weight_angle * l2_distant), torch.mean(l2_distant)
