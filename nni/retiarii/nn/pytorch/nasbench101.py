import numpy as np
import logging
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .api import InputChoice, ValueChoice, LayerChoice
from .utils import generate_new_label, get_fixed_dict
from ...mutator import Mutator

_logger = logging.getLogger(__name__)


def compute_vertex_channels(input_channels, output_channels, matrix):
    """
    This is (almost) copied from the original NAS-Bench-101 implementation.

    Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of channels at each interior vertex.
    Interior vertices have the same number of channels as the max of the channels of the vertices it feeds into.
    The output channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to compensate.

    Parameters
    ----------
    in_channels : int
        input channels count.
    output_channels : int
        output channel count.
    matrix : np.ndarray
        adjacency matrix for the module (pruned by model_spec).

    Returns
    -------
    list of int
        list of channel counts, in order of the vertices.
    """

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
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


class _NasBench101CellFixed(nn.Module):
    """
    The fixed version of NAS-Bench-101 Cell, used in python-version execution engine.
    """

    def __init__(self, operations: List[Callable[[int], nn.Module]],
                 adjacency_list: List[List[int]],
                 in_features: int, out_features: int, num_nodes: int,
                 projection: Callable[[int, int], nn.Module]):
        super().__init__()

        assert num_nodes == len(operations) + 2 == len(adjacency_list)

        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.connection_matrix = self._build_connection_matrix(adjacency_list)
        self.hidden_features = compute_vertex_channels(in_features, out_features, self.connection_matrix)

        self.projections = nn.ModuleList([nn.Identity()])
        self.op = nn.ModuleList([nn.Identity()])
        self.inputs = nn.ModuleList([nn.Identity()])
        for i in range(1, num_nodes):
            self.projections.append(projection(in_features, self.hidden_features[i]))

        for i in range(1, num_nodes - 1):
            self.op.append(operations[i - 1](self.hidden_features[i]))

    def _build_connection_matrix(self, adjacency_list):
        connections = np.zeros((self.num_nodes, self.num_nodes), dtype='int')
        for i, lst in enumerate(adjacency_list):
            assert all([0 <= k < i for k in lst])
            for k in lst:
                connections[k, i] = 1
        assert len(adjacency_list[-1]) >= 1, 'Last node must have inputs.'
        return connections

    def forward(self, inputs):
        tensors = [inputs]
        for t in range(1, self.num_nodes - 1):

            # Create interior connections, truncating if necessary
            add_in = [truncate(tensors[src], self.hidden_features[t])
                      for src in range(1, t) if self.connection_matrix[src, t]]

            # Create add connection from projected input
            if self.connection_matrix[0, t]:
                add_in.append(self.projections[t](tensors[0]))

            if len(add_in) == 1:
                vertex_input = add_in[0]
            else:
                vertex_input = sum(add_in)

            # Perform op at vertex t
            vertex_out = self.ops[t](vertex_input)
            tensors.append(vertex_out)

        # Construct final output tensor by concating all fan-in and adding input.
        if np.sum(self.connection_matrix[:, -1]) == 1:
            src = np.where(self.connection_matrix[:, -1] == 1)[0][0]
            return self.projections[-1](tensors[0]) if src == 0 else tensors[src]

        outputs = torch.cat([tensors[src] for src in range(1, self.num_nodes - 1) if self.connection_matrix[src, -1]], 1)
        if self.connection_matrix[0, -1]:
            outputs += self.projections[-1](tensors[0])
        assert outputs.size(1) == self.out_features
        return outputs


class NasBench101Cell(nn.Module):
    """
    Cell structure that is proposed in NAS-Bench-101 [nasbench101]_ .

    The space of this cell architecture consists of all possible directed acyclic graphs on no more than ``max_num_nodes`` nodes,
    where each possible node (other than IN and OUT) has one of ``op_candidates``, representing the corresponding operation.
    Edges connecting the nodes can be no more than ``max_num_edges``. 
    To align with the paper settings, two vertices specially labeled as operation IN and OUT, are also counted into
    ``max_num_nodes`` in our implementaion, the default value of ``max_num_nodes`` is 7 and ``max_num_edges`` is 9.

    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be `[N, C_{out}, *]`. The shape
    of each hidden nodes will be first automatically computed, depending on the cell structure. Each of the ``op_candidates``
    should be a callable that accepts computed ``num_features`` and returns a ``Module``. For example,

    .. code-block:: python

        def conv_bn_relu(num_features):
            return nn.Sequential(
                nn.Conv2d(num_features, num_features, 1),
                nn.BatchNorm2d(num_features),
                nn.ReLU()
            )

    The output of each node is the sum of its input node feed into its operation, except for the last node (output node),
    which is the concatenation of its input *hidden* nodes, adding the *IN* node (if IN and OUT are connected).

    When input tensor is added with any other tensor, there could be shape mismatch. Therefore, a projection transformation
    is needed to transform the input tensor. In paper, this is simply a Conv1x1 followed by BN and ReLU. The ``projection``
    parameters accepts ``in_features`` and ``out_features``, returns a ``Module``. This parameter has no default value,
    as we hold no assumption that users are dealing with images. An example for this parameter is,

    .. code-block:: python

        def projection_fn(in_features, out_features):
            return nn.Conv2d(in_features, out_features, 1)

    References
    ----------
    .. [nasbench101] Ying, Chris, et al. "Nas-bench-101: Towards reproducible neural architecture search."
        International Conference on Machine Learning. PMLR, 2019.
    """

    def __new__(cls, op_candidates: List[Callable[[int], nn.Module]],
                in_features: int, out_features: int, projection: Callable[[int, int], nn.Module],
                max_num_nodes: int = 7, max_num_edges: int = 9, label: Optional[str] = None):
        try:
            label = generate_new_label(label)
            selected = get_fixed_dict(label + '/')
            num_nodes = selected[f'{label}/num_nodes']
            adjacency_list = [selected[f'{label}/input_{i}'] for i in range(1, num_nodes)]
            assert sum([len(e) for e in adjacency_list]) <= max_num_edges, f'Expected {max_num_edges} edges, found: {adjacency_list}'
            return _NasBench101CellFixed(
                [op_candidates[selected[f'{label}/op_{i}']] for i in range(1, num_nodes - 1)],
                adjacency_list, in_features, out_features, num_nodes, projection)
        except AssertionError:
            return super().__new__(cls)

    def __init__(self, op_candidates: List[Callable[[int], nn.Module]],
                 in_features: int, out_features: int, projection: Callable[[int, int], nn.Module],
                 max_num_nodes: int = 7, max_num_edges: int = 9, label: Optional[str] = None):

        super().__init__()
        self._label = generate_new_label(label)
        num_vertices_prior = [2 ** i for i in range(2, max_num_nodes + 1)]
        num_vertices_prior = (np.array(num_vertices_prior) / sum(num_vertices_prior)).tolist()
        self.num_nodes = ValueChoice(list(range(2, max_num_nodes + 1)),
                                     prior=num_vertices_prior,
                                     label=f'{self._label}/num_nodes')
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges

        # this is only for input validation and instantiating enough layer choice and input choice
        self.hidden_features = out_features

        self.projections = nn.ModuleList([nn.Identity()])
        self.ops = nn.ModuleList([nn.Identity()])
        self.inputs = nn.ModuleList([nn.Identity()])
        for _ in range(1, max_num_nodes):
            self.projections.append(projection(in_features, self.hidden_features))
        for i in range(1, max_num_nodes):
            if i < max_num_nodes - 1:
                self.ops.append(LayerChoice([op(self.hidden_features) for op in op_candidates],
                                            label=f'{self._label}/op_{i}'))
            self.inputs.append(InputChoice(i, None, label=f'{self._label}/input_{i}'))

    @property
    def label(self):
        return self._label

    def forward(self, x):
        # This is a dummy forward and actually not used
        tensors = [x]
        for i in range(1, self.max_num_nodes):
            node_input = self.inputs([self.projections[i](tensors[0])] + [t for t in tensors[1:]])
            if i < self.max_num_nodes - 1:
                node_output = self.ops[i](node_input)
            else:
                node_output = node_input
            tensors.append(node_output)
        return tensors[-1]


class NasBench101Mutator(Mutator):
    # for validation purposes
    pass
