import logging
import math

import numpy as np
import torch
import torch.nn as nn

from sdk.mutators import Mutator
from sdk.translate_code import gen_pytorch_graph
from .blocks import Conv3x3BnRelu, Conv1x1BnRelu, MaxPool3x3, ConvBnRelu

OP_MAP = {
    'conv3x3': Conv3x3BnRelu,
    'conv1x1': Conv1x1BnRelu,
    'maxpool': MaxPool3x3
}


class Projection(Conv1x1BnRelu):
    pass


_logger = logging.getLogger(__name__)


def truncate(inputs, channels):
    input_channels = inputs.size(1)
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def compute_vertex_channels(output_channels, matrix):
    """
    Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = -1  # input channels can be inferred
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going backwards.
    # (num_vertices - 2) index skipped because it only connects to output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    _logger.debug('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


class Cell(nn.Module):

    def __init__(self, operations, connections, in_channels, out_channels):
        super(Cell, self).__init__()

        self.num_vertices = len(operations)
        self.operations = operations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.connections = connections
        connections = np.array(self.connections)
        assert len(connections) == self.num_vertices
        in_degree = np.sum(connections[1:], axis=0)
        self.vertex_channels = compute_vertex_channels(out_channels, connections)
        self.vertex_channels[0] = in_channels

        self.projections = nn.ModuleList([nn.Identity()])
        self.op = nn.ModuleList([nn.Identity()])
        for i in range(1, self.num_vertices):
            if connections[0, i]:
                self.projections.append(Projection(self.vertex_channels[0], self.vertex_channels[i]))
            else:
                self.projections.append(nn.Identity())
        for i in range(1, self.num_vertices - 1):
            self.op.append(OP_MAP[operations[i]](self.vertex_channels[i], self.vertex_channels[i]))

    def forward(self, inputs):
        connections = np.array(self.connections)
        tensors = [inputs]
        final_concat_in = []
        for t in range(1, self.num_vertices - 1):
            add_in = [truncate(tensors[src], self.vertex_channels[t])
                      for src in range(1, t) if connections[src, t]]
            if connections[0, t]:
                add_in.append(self.projections[t](tensors[0]))
            if len(add_in) == 1:
                vertex_input = add_in[0]
            else:
                vertex_input = sum(add_in)
            vertex_value = self.op[t](vertex_input)
            tensors.append(vertex_value)
            if connections[t, self.num_vertices - 1]:
                final_concat_in.append(vertex_value)
        if not final_concat_in:
            assert connections[0, self.num_vertices - 1]
            outputs = self.op[-1](tensors[0])
        else:
            if len(final_concat_in) == 1:
                outputs = final_concat_in[0]
            else:
                outputs = torch.cat(final_concat_in, 1)
            if connections[0, self.num_vertices - 1]:
                outputs += self.projections[self.num_vertices - 1](inputs)
        return outputs


class Nb101Network(nn.Module):
    def __init__(self, stem_out_channels=128, num_stacks=3,
                 num_modules_per_stack=3, dropout_rate=0.5, num_labels=10):
        super().__init__()

        # initial stem convolution
        self.stem_conv = Conv3x3BnRelu(3, stem_out_channels)
        self.aux_pos = -1

        layers = []
        in_channels = out_channels = stem_out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                layers.append(downsample)
                out_channels *= 2
            for _ in range(num_modules_per_stack):
                cell = Cell(['input'] + ['conv3x3'] * 5 + ['output'],
                            [[int(i + 1 == j) for j in range(7)] for i in range(7)],
                            in_channels, out_channels)
                layers.append(cell)
                in_channels = out_channels

        self.features = nn.ModuleList(layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(out_channels, num_labels)

    def forward(self, x):
        bs = x.size(0)
        out = self.stem_conv(x)
        for layer in self.features:
            out = layer(out)
        out = self.gap(out).view(bs, -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


class CellMutator(Mutator):
    def __init__(self, targets: 'List[str]'):
        self.targets = targets

    def retrieve_targeted_graphs(self, graph: 'Graph') -> 'List[Graph]':
        return [graph.find_node(t) for t in self.targets]

    def _validate(self, connections):
        connections = np.array(connections)
        if np.sum(connections) > 9:
            return False
        connections = np.linalg.matrix_power(connections + np.eye(len(connections)), len(connections))
        return np.all(connections[:, -1]) and np.all(connections[0, :])

    def mutate(self, graph):
        while True:
            # num_vertices = 2 is not supported yet
            num_vertices = self.choice(list(range(3, 7)))
            operations = ['input'] + [self.choice(list(OP_MAP.keys())) for _ in range(num_vertices - 2)] + ['output']
            connections = [[0 if i >= j else self.choice([0, 1]) for j in range(num_vertices)]
                           for i in range(num_vertices)]
            if self._validate(connections):
                break
        for target_node in self.retrieve_targeted_graphs(graph):
            target_node.update_operation(None, operations=operations, connections=connections)


def nasbench101():
    model = Nb101Network()

    cells = []
    for name, module in model.named_modules():
        if isinstance(module, Cell):
            cells.append(name)

    model_graph = gen_pytorch_graph(model, dummy_input=(torch.randn(1, 3, 224, 224),),
                                    collapsed_nodes={name: 'Cell' for name in cells})
    mutators = [CellMutator(cells)]
    return model_graph, mutators
