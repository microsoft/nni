
import torch
import torch.nn as nn

import nni.nas.pytorch as nas
from .cnn_ops import ops, Zero, FactorizedReduce, StdConv, Identity
from nni.nas.pytorch.modules import RankedModule


class CnnCell(RankedModule):
    """
    Cell for search.
    """

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        """
        Initialization a search cell.

        Parameters
        ----------
        n_nodes: int
            Number of nodes in current DAG.
        channels_pp: int
            Number of output channels from previous previous cell.
        channels_p: int
            Number of output channels from previous cell.
        channels: int
            Number of channels that will be used in the current DAG.
        reduction_p: bool
            Flag for whether the previous cell is reduction cell or not.
        reduction: bool
            Flag for whether the current cell is reduction cell or not.
        """
        super(CnnCell, self).__init__(rank=1)
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(
                channels_pp, channels, affine=False)
        else:
            self.preproc0 = StdConv(
                channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(self.n_nodes):
            # self.mutable_ops.append(nn.ModuleList())
            for i in range(2 + depth):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and i < 2 else 1
                m_ops = []
                for primitive in ops.PRIMITIVES:
                    op = ops.OPS_TABLE[primitive](channels, stride, False)
                    m_ops.append(op)
                op = nas.mutables.LayerChoice(m_ops,
                                              key="r{}_d{}_i{}".format(reduction, depth, i))
                self.mutable_ops.append(op)

    def forward(self, s0, s1):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for ops in self.mutable_ops:
            assert len(ops) == len(tensors)
            cur_tensor = sum(op(tensor) for op, tensor in zip(ops, tensors))
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output
