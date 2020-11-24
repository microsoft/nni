# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutables import LayerChoice, InputChoice


class MutableOp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(3, 120, kernel_size, padding=kernel_size // 2)
        self.nested_mutable = InputChoice(n_candidates=10)

    def forward(self, x):
        return self.conv(x)


class NestedSpace(nn.Module):
    # this doesn't pass tests
    def __init__(self, test_case):
        super().__init__()
        self.test_case = test_case
        self.conv1 = LayerChoice([MutableOp(3), MutableOp(5)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(120, 10)

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.gap(x).view(bs, -1)
        x = self.fc(x)
        return x
