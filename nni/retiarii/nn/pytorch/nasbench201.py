from collections import OrderedDict
from copy import deepcopy

import torch.nn as nn

from common.searchspace import SearchSpace, MixedOp
from configs import NasBench201Config
from .ops import PRIMITIVES, OPS, ResNetBasicblock


class NasBench201Cell(nn.Module):
    NUM_NODES = 4

    def __init__(self, op_candidates, num_tensors, num_features):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_nodes):
            for j in range(i):


        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                op_choices = OrderedDict([(op, OPS[op](C_in, C_out, stride if j == 0 else 1)) for op in PRIMITIVES])
                node_ops.append(MixedOp(f'{j}_{i}', op_choices))
            self.layers.append(node_ops)
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id
        self.stride = stride

    def forward(self, inputs):
        nodes = [inputs]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class NasBench201(SearchSpace):
    def __init__(self, config: NasBench201Config):
        super(NasBench201, self).__init__()
        self.channels = C = config.stem_out_channels
        self.num_modules = N = config.num_modules_per_stack
        self.num_labels = config.num_labels

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = Cell(i, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits