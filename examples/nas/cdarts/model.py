# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import ops
import numpy as np
from nni.nas.pytorch import mutables
from utils import parse_results
from aux_head import DistillHeadCIFAR, DistillHeadImagenet, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(mutables.LayerChoice([ops.OPS[k](channels, stride, False) for k in ops.PRIMITIVES],
                                                 key=choice_keys[-1]))
        self.drop_path = ops.DropPath()
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class Cell(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output


class Model(nn.Module):

    def __init__(self, dataset, n_layers, in_channels=3, channels=16, n_nodes=4, retrain=False, shared_modules=None):
        super().__init__()
        assert dataset in ["cifar10", "imagenet"]
        self.dataset = dataset
        self.input_size = 32 if dataset == "cifar" else 224
        self.in_channels = in_channels
        self.channels = channels
        self.n_nodes = n_nodes
        self.aux_size = {2 * n_layers // 3: self.input_size // 4}
        if dataset == "cifar10":
            self.n_classes = 10
            self.aux_head_class = AuxiliaryHeadCIFAR if retrain else DistillHeadCIFAR
            if not retrain:
                self.aux_size = {n_layers // 3: 6, 2 * n_layers // 3: 6}
        elif dataset == "imagenet":
            self.n_classes = 1000
            self.aux_head_class = AuxiliaryHeadImageNet if retrain else DistillHeadImagenet
            if not retrain:
                self.aux_size = {n_layers // 3: 6, 2 * n_layers // 3: 5}
        self.n_layers = n_layers
        self.aux_head = nn.ModuleDict()
        self.ensemble_param = nn.Parameter(torch.rand(len(self.aux_size) + 1) / (len(self.aux_size) + 1)) \
            if not retrain else None

        stem_multiplier = 3 if dataset == "cifar" else 1
        c_cur = stem_multiplier * self.channels
        self.shared_modules = {}  # do not wrap with ModuleDict
        if shared_modules is not None:
            self.stem = shared_modules["stem"]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_cur)
            )
            self.shared_modules["stem"] = self.stem

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        aux_head_count = 0
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            if i in self.aux_size:
                if shared_modules is not None:
                    self.aux_head[str(i)] = shared_modules["aux" + str(aux_head_count)]
                else:
                    self.aux_head[str(i)] = self.aux_head_class(c_cur_out, self.aux_size[i], self.n_classes)
                    self.shared_modules["aux" + str(aux_head_count)] = self.aux_head[str(i)]
                aux_head_count += 1
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, self.n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        outputs = []

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if str(i) in self.aux_head:
                outputs.append(self.aux_head[str(i)](s1))

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        outputs.append(logits)

        if self.ensemble_param is None:
            assert len(outputs) == 2
            return outputs[1], outputs[0]
        else:
            em_output = torch.cat([(e * o) for e, o in zip(F.softmax(self.ensemble_param, dim=0), outputs)], 0)
            return logits, em_output

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p

    def plot_genotype(self, results, logger):
        genotypes = parse_results(results, self.n_nodes)
        logger.info(genotypes)
        return genotypes
