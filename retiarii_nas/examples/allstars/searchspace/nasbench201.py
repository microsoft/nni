from copy import deepcopy

import torch.nn as nn
from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph

from .blocks import *


OPS = {
    'none': lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'avg'),
    'max_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'max'),
    'nor_conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
    'nor_conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
    'skip_connect': lambda C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride),
}

PRIMITIVES = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


class Cell(nn.Module):
    num_nodes = 4

    def __init__(self, cell_id, op_dict, in_dim, out_dim, stride):
        super(Cell, self).__init__()

        self.op_dict = op_dict
        self.layers = nn.ModuleList()
        for i in range(self.num_nodes):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                node_ops.append(OPS[op_dict[f'{j}_{i}']](in_dim, out_dim, stride if j == 0 else 1))
            self.layers.append(node_ops)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cell_id = cell_id
        self.stride = stride

    def forward(self, inputs):
        nodes = [inputs]
        for i in range(1, self.num_nodes):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class Nb201Network(nn.Module):
    def __init__(self, stem_out_channels=16, num_modules_per_stack=5):
        super(Nb201Network, self).__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = 10

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=0.1)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = Cell(i, {f'{i}_{j}': PRIMITIVES[-1] for i in range(4) for j in range(i + 1, 4)},
                            C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=0.1),
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


class CellMutator(Mutator):
    def __init__(self, targets: 'List[str]'):
        self.targets = targets

    def retrieve_targeted_graphs(self, graph: 'Graph') -> 'List[Graph]':
        return [graph.find_node(t) for t in self.targets]

    def mutate(self, graph):
        target_nodes = self.retrieve_targeted_graphs(graph)
        operations = {f'{i}_{j}': self.choice(list(OPS.keys())) for i in range(4) for j in range(i + 1, 4)}
        for target_node in target_nodes:
            target_node.update_operation(None, op_dict=operations)


def nasbench201():
    model = Nb201Network()

    cells = []
    for name, module in model.named_modules():
        if isinstance(module, Cell):
            cells.append(name)

    model_graph = gen_pytorch_graph(model, dummy_input=(torch.randn(1, 3, 32, 32),),
                                    collapsed_nodes={name: 'Cell' for name in cells})
    mutators = [CellMutator(cells)]
    return model_graph, mutators
