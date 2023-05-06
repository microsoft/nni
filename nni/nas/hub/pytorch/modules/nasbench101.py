# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

__all__ = ['NasBench101Cell']

import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Union, Dict, Tuple, Callable, Iterable, Any, cast

import numpy as np
import torch
import torch.nn as nn

from nni.mutable import (
    Constraint, Categorical, CategoricalMultiple, Constraint, ConstraintViolation,
    Mutable, LabeledMutable, Sample, SampleValidationError,
    auto_label, label_scope
)
from nni.mutable.mutable import _mutable_equal
from nni.nas.nn.pytorch import InputChoice, LayerChoice, MutableModule

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


def prune(matrix, ops) -> Tuple[np.ndarray, List[Union[str, Callable[[int], nn.Module]]]]:
    """
    Prune the extraneous parts of the graph.

    General procedure:

    1. Remove parts of graph not connected to input.
    2. Remove parts of graph not connected to output.
    3. Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(matrix)[0]

    # calculate the connection matrix within V number of steps.
    connections = np.linalg.matrix_power(matrix + np.eye(num_vertices), num_vertices)

    visited_from_input = set([i for i in range(num_vertices) if connections[0, i]])
    visited_from_output = set([i for i in range(num_vertices) if connections[i, -1]])

    # Any vertex that isn't connected to both input and output is extraneous to the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    if len(extraneous) > num_vertices - 2:
        raise ConstraintViolation('Non-extraneous graph is less than 2 vertices, '
                                  'the input is not connected to the output and the spec is invalid.')

    matrix = np.delete(matrix, list(extraneous), axis=0)
    matrix = np.delete(matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
        del ops[index]
    return matrix, ops


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
        return inputs[:, :channels]


class _NasBench101CellFixed(nn.Module):
    """
    The fixed version of NAS-Bench-101 Cell, used in python-version execution engine.
    """

    def __init__(self, operations: List[Callable[[int], nn.Module]],
                 adjacency_list: List[List[int]],
                 in_features: int, out_features: int, num_nodes: int,
                 projection: Callable[[int, int], nn.Module]):
        super().__init__()

        assert num_nodes == len(operations) + 2 == len(adjacency_list) + 1

        raw_operations: List[Union[str, Callable[[int], nn.Module]]] = list(operations)
        del operations  # operations is no longer needed. Delete it to avoid misuse

        # add psuedo nodes
        raw_operations.insert(0, 'IN')
        raw_operations.append('OUT')

        self.connection_matrix = self.build_connection_matrix(adjacency_list, num_nodes)
        del num_nodes  # raw number of nodes is no longer used

        self.connection_matrix, self.operations = prune(self.connection_matrix, raw_operations)

        self.hidden_features = compute_vertex_channels(in_features, out_features, self.connection_matrix)

        self.num_nodes = len(self.connection_matrix)
        self.in_features = in_features
        self.out_features = out_features
        _logger.info('Prund number of nodes: %d', self.num_nodes)
        _logger.info('Pruned connection matrix: %s', str(self.connection_matrix))

        self.projections = nn.ModuleList([nn.Identity()])
        self.ops = nn.ModuleList([nn.Identity()])
        for i in range(1, self.num_nodes):
            self.projections.append(projection(in_features, self.hidden_features[i]))

        for i in range(1, self.num_nodes - 1):
            operation = cast(Callable[[int], nn.Module], self.operations[i])
            self.ops.append(operation(self.hidden_features[i]))

    @staticmethod
    def build_connection_matrix(adjacency_list, num_nodes):
        adjacency_list = [[]] + adjacency_list  # add adjacency for first node
        connections = np.zeros((num_nodes, num_nodes), dtype='int')
        for i, lst in enumerate(adjacency_list):
            assert all([0 <= k < i for k in lst])
            for k in lst:
                connections[k, i] = 1
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


class NasBench101Cell(MutableModule):
    """
    Cell structure that is proposed in NAS-Bench-101.

    Proposed by `NAS-Bench-101: Towards Reproducible Neural Architecture Search <http://proceedings.mlr.press/v97/ying19a/ying19a.pdf>`__.

    This cell is usually used in evaluation of NAS algorithms because there is a "comprehensive analysis" of this search space
    available, which includes a full architecture-dataset that "maps 423k unique architectures to metrics
    including run time and accuracy". You can also use the space in your own space design, in which scenario it should be possible
    to leverage results in the benchmark to narrow the huge space down to a few efficient architectures.

    The space of this cell architecture consists of all possible directed acyclic graphs on no more than ``max_num_nodes`` nodes,
    where each possible node (other than IN and OUT) has one of ``op_candidates``, representing the corresponding operation.
    Edges connecting the nodes can be no more than ``max_num_edges``.
    To align with the paper settings, two vertices specially labeled as operation IN and OUT, are also counted into
    ``max_num_nodes`` in our implementation, the default value of ``max_num_nodes`` is 7 and ``max_num_edges`` is 9.

    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be :math:`[N, C_{out}, *]`. The shape
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

    Parameters
    ----------
    op_candidates : list of callable
        Operation candidates. Each should be a function accepts number of feature, returning nn.Module.
    in_features : int
        Input dimension of cell.
    out_features : int
        Output dimension of cell.
    projection : callable
        Projection module that is used to preprocess the input tensor of the whole cell.
        A callable that accept input feature and output feature, returning nn.Module.
    max_num_nodes : int
        Maximum number of nodes in the cell, input and output included. At least 2. Default: 7.
    max_num_edges : int
        Maximum number of edges in the cell. Default: 9.
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    Warnings
    --------
    :class:`NasBench101Cell` is not supported for graph-based model format.
    It's also not supported by most one-shot algorithms currently.
    """

    @staticmethod
    def _make_dict(x):
        if isinstance(x, list):
            return OrderedDict([(str(i), t) for i, t in enumerate(x)])
        return OrderedDict(x)

    @classmethod
    def create_fixed_module(cls, sample: dict,
                            op_candidates: Union[Dict[str, Callable[[int], nn.Module]], List[Callable[[int], nn.Module]]],
                            in_features: int, out_features: int, projection: Callable[[int, int], nn.Module],
                            max_num_nodes: int = 7, max_num_edges: int = 9, label: Optional[Union[str, label_scope]] = None):
        with (label if isinstance(label, label_scope) else label_scope(label)):
            # Freeze number of nodes.
            num_nodes = cls._num_nodes_discrete(max_num_nodes)
            num_nodes_frozen = num_nodes.freeze(sample)

            # Freeze operations.
            op_candidates = cls._make_dict(op_candidates)
            op_choices = [cls._op_discrete(op_candidates, i) for i in range(1, num_nodes_frozen - 1)]
            op_frozen = [op_candidates[op.freeze(sample)] for op in op_choices]

            # Freeze inputs.
            input_choices = [cls._input_discrete(i) for i in range(1, num_nodes_frozen)]
            input_frozen = [inp.freeze(sample) for inp in input_choices]

            # Check constraint.
            NasBench101CellConstraint(max_num_edges, num_nodes, op_choices, input_choices).freeze(sample)

            return _NasBench101CellFixed(op_frozen, input_frozen, in_features, out_features, num_nodes_frozen, projection)

    # These should belong to LayerChoice and InputChoice.
    # But we rewrite them so that we don't have to create the layers in fixed mode.
    @staticmethod
    def _op_discrete(op_candidates: Dict[str, Callable[[int], nn.Module]], index: int):
        return Categorical(list(op_candidates), label=f'op{index}')

    @staticmethod
    def _input_discrete(index: int):
        return CategoricalMultiple(range(index), label=f'input{index}')

    @staticmethod
    def _num_nodes_discrete(max_num_nodes: int):
        num_vertices_prior = [2 ** i for i in range(2, max_num_nodes + 1)]
        num_vertices_prior = (np.array(num_vertices_prior) / sum(num_vertices_prior)).tolist()
        return Categorical(range(2, max_num_nodes + 1), weights=num_vertices_prior, label='num_nodes')

    def __init__(self, op_candidates: Union[Dict[str, Callable[[int], nn.Module]], List[Callable[[int], nn.Module]]],
                 in_features: int, out_features: int, projection: Callable[[int, int], nn.Module],
                 max_num_nodes: int = 7, max_num_edges: int = 9, label: Optional[Union[str, label_scope]] = None):

        super().__init__()
        if isinstance(label, label_scope):
            self._scope = label
        else:
            self._scope = label_scope(label)

        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.in_features = in_features
        self.out_features = out_features

        self.op_candidates = self._make_dict(op_candidates)
        self.projection = projection

        with self._scope:
            self.num_nodes = self._num_nodes_discrete(max_num_nodes)

            # Fake hidden features that is large enough.
            self.hidden_features = out_features

            self.projections = nn.ModuleList([nn.Identity()])
            self.ops = nn.ModuleList([nn.Identity()])
            self.inputs = nn.ModuleList([nn.Identity()])
            for _ in range(1, max_num_nodes):
                self.projections.append(projection(in_features, self.hidden_features))

            # The underlying `Categorical` of ops and inputs
            op_inner: List[Categorical] = []
            input_inner: List[CategoricalMultiple] = []
            for i in range(1, max_num_nodes):
                # Create layer
                if i < max_num_nodes - 1:
                    layer = LayerChoice({k: op(self.hidden_features) for k, op in self.op_candidates.items()}, label=f'op{i}')
                    op_inner.append(self._op_discrete(self.op_candidates, i))
                    assert layer.choice.equals(op_inner[-1])  # Make sure the choice is the same
                    self.ops.append(layer)
                # Create input
                inp = InputChoice(i, None, label=f'input{i}')
                input_inner.append(self._input_discrete(i))
                assert inp.choice.equals(input_inner[-1])  # Make sure the input choice is correct
                self.inputs.append(inp)

            self.constraint = NasBench101CellConstraint(self.max_num_edges, self.num_nodes, op_inner, input_inner)
            self.add_mutable(self.constraint)

    @property
    def label(self):
        return self._scope.name

    def freeze(self, sample: Dict[str, Any]) -> nn.Module:
        return self.create_fixed_module(sample, self.op_candidates, self.in_features, self.out_features, self.projection,
                                        self.max_num_nodes, self.max_num_edges, self._scope)

    def forward(self, x):
        """Forward of NasBench101Cell is unimplemented."""
        raise NotImplementedError(
            'The forward of NasBench101Cell should never be called directly. '
            'Either freeze it or use it in a search algorithm.'
        )


class NasBench101CellConstraint(Constraint):
    def __init__(
        self,
        max_num_edges: int,
        num_nodes: Categorical[int],
        operations: List[Categorical],
        inputs: List[CategoricalMultiple],
    ):
        self.label = auto_label('final')
        self.max_num_edges = max_num_edges
        self.num_nodes = num_nodes
        self.operations = operations
        self.inputs = inputs

    def equals(self, other: Any) -> bool:
        return isinstance(other, NasBench101CellConstraint) and \
            self.label == other.label and \
            self.max_num_edges == other.max_num_edges and \
            self.num_nodes.equals(other.num_nodes) and \
            _mutable_equal(self.operations, other.operations) and \
            _mutable_equal(self.inputs, other.inputs)

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        yield from self.num_nodes.leaf_mutables(is_leaf)
        for operator in self.operations:
            yield from operator.leaf_mutables(is_leaf)
        for inp in self.inputs:
            yield from inp.leaf_mutables(is_leaf)
        yield self

    def check_contains(self, sample: Sample) -> Optional[SampleValidationError]:
        # Check num_nodes
        err = self.num_nodes.check_contains(sample)
        if err is not None:
            err.paths.append('num_nodes')
            return err
        num_nodes = self.num_nodes.freeze(sample)  # must succeed
        assert num_nodes >= 2

        # Check connections
        adjacency_list: List[List[int]] = []
        for i, inp in enumerate(self.inputs[:num_nodes - 1], start=1):
            err = inp.check_contains(sample)
            if err is not None:
                err.paths.append(f'input{i}')
                return err
            adjacency_list.append(inp.freeze(sample))
        if sum([len(e) for e in adjacency_list]) > self.max_num_edges:
            return ConstraintViolation(f'Expected at most {self.max_num_edges} edges, found: {adjacency_list}')
        matrix = _NasBench101CellFixed.build_connection_matrix(adjacency_list, num_nodes)

        # Check operations
        operations: List[str] = ['IN']
        for i, op in enumerate(self.operations[:num_nodes - 2], start=1):
            err = op.check_contains(sample)
            if err is not None:
                err.paths.append(f'op{i}')
                return err
            operations.append(op.freeze(sample))
        operations.append('OUT')
        if len(operations) != len(matrix):
            raise RuntimeError('The number of operations does not match the number of nodes')

        try:
            cur_matrix, cur_operations = prune(matrix, operations)
        except ConstraintViolation as err:
            err.paths.append('prune')
            return err

        # Maintain a clean copy of what nasbench101 cell looks like.
        # Modifies sample in-place. A bit hacky here.
        rv: Dict[str, Any] = {}
        for i in range(1, len(cur_matrix)):
            if i + 1 < len(cur_matrix):
                rv[f'op{i}'] = cur_operations[i]
            rv[f'input{i}'] = [k for k in range(i) if cur_matrix[k, i]]
        sample[self.label] = rv

    def freeze(self, sample: Sample) -> Dict[str, Any]:
        self.validate(sample)
        assert self.label in sample
        return sample[self.label]
