# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn
from nni.nas.pytorch import mutables

from .darts_ops import PoolBN, SepConv, DilConv, FactorizedReduce, DropPath, StdConv


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        """
        builtin Darts Node structure

        Parameters
        ---
        node_id: str
        num_prev_nodes: int
            the number of previous nodes in this cell
        channels: int
            output channels
        num_downsample_connect: int
            downsample the input node if this cell is reduction cell
        """
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                mutables.LayerChoice(OrderedDict([
                    ("maxpool", PoolBN('max', channels, 3, stride, 1, affine=False)),
                    ("avgpool", PoolBN('avg', channels, 3, stride, 1, affine=False)),
                    ("skipconnect",
                     nn.Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=False)),
                    ("sepconv3x3", SepConv(channels, channels, 3, stride, 1, affine=False)),
                    ("sepconv5x5", SepConv(channels, channels, 5, stride, 2, affine=False)),
                    ("dilconv3x3", DilConv(channels, channels, 3, stride, 2, 2, affine=False)),
                    ("dilconv5x5", DilConv(channels, channels, 5, stride, 4, 2, affine=False))
                ]), key=choice_keys[-1]))
        self.drop_path = DropPath()
        self.input_switch = mutables.InputChoice(choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id))

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class DartsCell(nn.Module):
    """
    Builtin Darts Cell structure. There are ``n_nodes`` nodes in one cell, in which the first two nodes' values are
    fixed to the results of previous previous cell and previous cell respectively. One node will connect all
    the nodes after with predefined operations in a mutable way. The last node accepts five inputs from nodes
    before and it concats all inputs in channels as the output of the current cell, and the number of output
    channels is ``n_nodes`` times ``channels``.

    Parameters
    ---
    n_nodes: int
        the number of nodes contained in this cell
    channels_pp: int
        the number of previous previous cell's output channels
    channels_p: int
        the number of previous cell's output channels
    channels: int
        the number of output channels for each node
    reduction_p: bool
        Is previous cell a reduction cell
    reduction: bool
        is current cell a reduction cell
    """
    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, pprev, prev):
        """
        Parameters
        ---
        pprev: torch.Tensor
            the output of the previous previous layer
        prev: torch.Tensor
            the output of the previous layer
        """
        tensors = [self.preproc0(pprev), self.preproc1(prev)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output
