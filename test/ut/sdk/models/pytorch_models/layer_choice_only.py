# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutables import LayerChoice


class LayerChoiceOnlySearchSpace(nn.Module):
    def __init__(self, test_case):
        super().__init__()
        self.test_case = test_case
        self.conv1 = LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)],
                                 return_mask=True)
        self.conv3 = nn.Conv2d(16, 16, 1)
        self.bn = nn.BatchNorm2d(16)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        bs = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x0, mask = self.conv2(x)
        self.test_case.assertEqual(mask.size(), torch.Size([2]))
        x1 = F.relu(self.conv3(x0))

        x = self.pool(self.bn(x1))
        self.test_case.assertEqual(mask.size(), torch.Size([2]))

        x = self.gap(x).view(bs, -1)
        x = self.fc(x)
        return x
