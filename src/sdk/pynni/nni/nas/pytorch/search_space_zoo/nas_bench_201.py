import torch.nn as nn
from nni.nas.pytorch.mutables import LayerChoice

from .nas_bench_201_ops import Pooling, ReLUConvBN, Zero, FactorizedReduce


class NASBench201Cell(nn.Module):
    """
    Builtin cell structure of NAS Bench 201. One cell contains four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes connect to all previous nodes with an edge that
    represents an operation chosen from a set to transform the tensor from the source node to the target node.
    Every node accepts all its inputs and adds them as its output. 

    Parameters
    ---
    configs: dict
        # TODO: repalce this with speific parameters
    cell_id: str
        the name of this cell
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    """
    def __init__(self, configs, cell_id, C_in, C_out, stride):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.ENABLE_VIS = False
        self.layers = nn.ModuleList()

        OPS = {
            "none": lambda configs, C_in, C_out, stride: Zero(configs, C_in, C_out, stride),
            "avg_pool_3x3": lambda configs, C_in, C_out, stride: Pooling(configs, C_in, C_out, stride, "avg"),
            "max_pool_3x3": lambda configs, C_in, C_out, stride: Pooling(configs, C_in, C_out, stride, "max"),
            "nor_conv_3x3": lambda configs, C_in, C_out, stride: ReLUConvBN(configs, C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
            "nor_conv_1x1": lambda configs, C_in, C_out, stride: ReLUConvBN(configs, C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
            "skip_connect": lambda configs, C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out else FactorizedReduce(configs, C_in, C_out, stride),
        }
        PRIMITIVES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                op_choices = [OPS[op](configs, C_in, C_out, stride if j == 0 else 1) for op in PRIMITIVES]
                node_ops.append(LayerChoice(op_choices, key="edge_%d_%d" % (j, i), reduction="mean"))
            self.layers.append(node_ops)
        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id

    def forward(self, inputs):
        """
        Parameters
        ---
        inputs: tensor
            the output of the previous layer
        """
        nodes = [inputs]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]
