from collections import OrderedDict
import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice

from .nasbench201_ops import Pooling, ReLUConvBN, Zero, FactorizedReduce


class NASBench201Cell(nn.Module):
    """
    Builtin cell structure of NAS Bench 201. One cell contains four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes connect to all previous nodes with an edge that
    represents an operation chosen from a set to transform the tensor from the source node to the target node.
    Every node accepts all its inputs and adds them as its output.

    Parameters
    ---
    cell_id: str
        the name of this cell
    C_in: int
        the number of input channels of the cell
    C_out: int
        the number of output channels of the cell
    stride: int
        stride of all convolution operations in the cell
    bn_affine: bool
        If set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell will have learnable affine parameters. Default: True
    bn_momentum: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell tracks the running mean and variance. Default: True
    """

    def __init__(self, cell_id, C_in, C_out, stride, bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()

        OPS = lambda layer_idx: OrderedDict([
            ("none", Zero(C_in, C_out, stride)),
            ("avg_pool_3x3", Pooling(C_in, C_out, stride if layer_idx == 0 else 1, bn_affine, bn_momentum,
                                     bn_track_running_stats)),
            ("conv_3x3", ReLUConvBN(C_in, C_out, 3, stride if layer_idx == 0 else 1, 1, 1, bn_affine, bn_momentum,
                                    bn_track_running_stats)),
            ("conv_1x1", ReLUConvBN(C_in, C_out, 1, stride if layer_idx == 0 else 1, 0, 1, bn_affine, bn_momentum,
                                    bn_track_running_stats)),
            ("skip_connect", nn.Identity() if stride == 1 and C_in == C_out
             else FactorizedReduce(C_in, C_out, stride if layer_idx == 0 else 1, bn_affine, bn_momentum,
                                   bn_track_running_stats))
        ])

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                node_ops.append(LayerChoice(OPS(j), key="%d_%d" % (j, i), reduction="mean"))
            self.layers.append(node_ops)
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id

    def forward(self, input): # pylint: disable=W0622
        """
        Parameters
        ---
        input: torch.tensor
            the output of the previous layer
        """
        nodes = [input]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]
