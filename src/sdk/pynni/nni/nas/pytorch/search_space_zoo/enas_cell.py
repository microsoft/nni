# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch import mutables
from .enas_ops import FactorizedReduce, StdConv, SepConvBN, Pool, ConvBranch, PoolBranch


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, 3, 2)
        )
        self.proj = nn.Sequential(
            StdConv(in_channels, 128),
            StdConv(128, 768)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, 10, bias=False)

    def forward(self, x):
        bs = x.size(0)
        x = self.pooling(x)
        x = self.proj(x)
        x = self.avg_pool(x).view(bs, -1)
        x = self.fc(x)
        return x


class Cell(nn.Module):
    def __init__(self, cell_name, prev_labels, channels):
        super().__init__()
        self.input_choice = mutables.InputChoice(choose_from=prev_labels, n_chosen=1, return_mask=True,
                                                 key=cell_name + "_input")
        self.op_choice = mutables.LayerChoice([
            SepConvBN(channels, channels, 3, 1),
            SepConvBN(channels, channels, 5, 2),
            Pool("avg", 3, 1, 1),
            Pool("max", 3, 1, 1),
            nn.Identity()
        ], key=cell_name + "_op")

    def forward(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers)
        cell_out = self.op_choice(chosen_input)
        return cell_out, chosen_mask


class Node(mutables.MutableScope):
    def __init__(self, node_name, prev_node_names, channels):
        super().__init__(node_name)
        self.cell_x = Cell(node_name + "_x", prev_node_names, channels)
        self.cell_y = Cell(node_name + "_y", prev_node_names, channels)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)
        return out_x + out_y, mask_x | mask_y


class Calibration(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.process = None
        if in_channels != out_channels:
            self.process = StdConv(in_channels, out_channels)

    def forward(self, x):
        if self.process is None:
            return x
        return self.process(x)


class ENASReductionLayer(nn.Module):
    """
    builtin EnasReductionLayer.

    Parameters
    ---
    in_channels_pp: int
        the number of previous previous layer's output channels
    in_channels_p: int
        the number of previous layer's output channels
    out_channels: int
        output channels of this layer
    """
    def __init__(self, in_channels_pp, in_channels_p, out_channels):
        super().__init__()
        self.reduce0 = FactorizedReduce(in_channels_pp, out_channels, affine=False)
        self.reduce1 = FactorizedReduce(in_channels_p, out_channels, affine=False)

    def forward(self, pprev, prev):
        return self.reduce0(pprev), self.reduce1(prev)


class ENASMicroLayer(nn.Module):
    """
    builin EnasMicroLayer. Micro search designs only one building block whose architecture is repeated throughout the final architecture.

    Parameters
    ---
    num_nodes: int
        the number of nodes contained in this layer
    in_channles_pp: int
        the number of previous previous layer's output channels
    in_channels_p: int
        the number of previous layer's output channels
    out_channels: int
        output channels of this layer
    reduction: bool
        is previous layer a reduction layer
    """
    def __init__(self, num_nodes, in_channels_pp, in_channels_p, out_channels, reduction):
        super().__init__()
        self.preproc0 = Calibration(in_channels_pp, out_channels)
        self.preproc1 = Calibration(in_channels_p, out_channels)

        self.num_nodes = num_nodes
        name_prefix = "reduce" if reduction else "normal"
        self.nodes = nn.ModuleList()
        node_labels = [mutables.InputChoice.NO_KEY, mutables.InputChoice.NO_KEY]
        for i in range(num_nodes):
            node_labels.append("{}_node_{}".format(name_prefix, i))
            self.nodes.append(Node(node_labels[-1], node_labels[:-1], out_channels))
        self.final_conv_w = nn.Parameter(torch.zeros(out_channels, self.num_nodes + 2, out_channels, 1, 1),
                                         requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.final_conv_w)

    def forward(self, pprev, prev):
        pprev_, prev_ = self.preproc0(pprev), self.preproc1(prev)

        prev_nodes_out = [pprev_, prev_]
        nodes_used_mask = torch.zeros(self.num_nodes + 2, dtype=torch.bool, device=prev.device)
        for i in range(self.num_nodes):
            node_out, mask = self.nodes[i](prev_nodes_out)
            nodes_used_mask[:mask.size(0)] |= mask.to(node_out.device)
            prev_nodes_out.append(node_out)

        unused_nodes = torch.cat([out for used, out in zip(nodes_used_mask, prev_nodes_out) if not used], 1)
        unused_nodes = F.relu(unused_nodes)
        conv_weight = self.final_conv_w[:, ~nodes_used_mask, :, :, :]
        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)
        out = F.conv2d(unused_nodes, conv_weight)
        return prev, self.bn(out)


class ENASMacroLayer(mutables.MutableScope):
    """
    builtin ENAS Marco Layer. With search space changing to layer level, controller decides what operation is employed and the previous layer to connect to for skip connections.
    The model is made up by the same layes but the choice of each layer may be different.

    Parameters
    ---
    key: str
        the name of this layer
    prev_labels: str
        names of all previous layers
    in_filters: int
        the number of input channels
    out_filters:
        the number of output channels
    """
    def __init__(self, key, prev_labels, in_filters, out_filters):
        super().__init__(key)
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mutable = mutables.LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)
        ])
        if len(prev_labels) > 0:
            self.skipconnect = mutables.InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        self.batch_norm = nn.BatchNorm2d(out_filters, affine=False)

    def forward(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
        return self.batch_norm(out)
