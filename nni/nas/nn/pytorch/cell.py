# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import warnings
from typing import Callable, Dict, List, Union, Optional, Tuple, Sequence, Union, cast
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn

from nni.mutable import Sample, label_scope
from nni.nas.space import model_context

from .choice import ChosenInputs, LayerChoice, InputChoice
from .base import MutableModule

_logger = logging.getLogger(__name__)


class _ListIdentity(nn.Identity):
    # workaround for torchscript
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return x


class _DefaultPostprocessor(nn.Module):
    # this is also a workaround for torchscript

    def forward(self, this_cell: torch.Tensor, prev_cell: List[torch.Tensor]) -> torch.Tensor:
        return this_cell


CellOpFactory = Callable[[int, int, Optional[int]], nn.Module]


def create_cell_op_candidates(
    op_candidates, node_index, op_index, chosen
) -> Tuple[Dict[str, nn.Module], bool]:
    has_factory = False

    # convert the complex type into the type that is acceptable to LayerChoice
    def convert_single_op(op):
        nonlocal has_factory

        if isinstance(op, nn.Module):
            return copy.deepcopy(op)
        elif callable(op):
            # Yes! It's using factory to create operations now.
            has_factory = True
            # FIXME: I don't know how to check whether we are in graph engine.
            return op(node_index, op_index, chosen)
        else:
            raise TypeError(f'Unrecognized type {type(op)} for op {op}')

    if isinstance(op_candidates, list):
        res = {str(i): convert_single_op(op) for i, op in enumerate(op_candidates)}
    elif isinstance(op_candidates, dict):
        res = {key: convert_single_op(op) for key, op in op_candidates.items()}
    elif callable(op_candidates):
        warnings.warn(f'Directly passing a callable into Cell is deprecated. Please consider migrating to list or dict.',
                      DeprecationWarning)
        res = op_candidates()
        has_factory = True
    else:
        raise TypeError(f'Unrecognized type {type(op_candidates)} for {op_candidates}')

    return res, has_factory


def preprocess_cell_inputs(num_predecessors: int, *inputs: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    if len(inputs) == 1 and isinstance(inputs[0], list):
        processed_inputs = list(inputs[0])  # shallow copy
    else:
        processed_inputs = cast(List[torch.Tensor], list(inputs))
    assert len(processed_inputs) == num_predecessors, 'The number of inputs must be equal to `num_predecessors`.'
    return processed_inputs


class Cell(MutableModule):
    """
    Cell structure that is popularly used in NAS literature.

    Find the details in:

    * `Neural Architecture Search with Reinforcement Learning <https://arxiv.org/abs/1611.01578>`__.
    * `Learning Transferable Architectures for Scalable Image Recognition <https://arxiv.org/abs/1707.07012>`__.
    * `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`__

    `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__
    is a good summary of how this structure works in practice.

    A cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from
    ``op_candidates``, and takes one input from previous nodes and predecessors. Predecessor means the input of cell.
    The output of cell is the concatenation of some of the nodes in the cell (by default all the nodes).

    Two examples of searched cells are illustrated in the figure below.
    In these two cells, ``op_candidates`` are series of convolutions and pooling operations.
    ``num_nodes_per_node`` is set to 2. ``num_nodes`` is set to 5. ``merge_op`` is ``loose_end``.
    Assuming nodes are enumerated from bottom to top, left to right,
    ``output_node_indices`` for the normal cell is ``[2, 3, 4, 5, 6]``.
    For the reduction cell, it's ``[4, 5, 6]``.
    Please take a look at this
    `review article <https://sh-tsang.medium.com/review-nasnet-neural-architecture-search-network-image-classification-23139ea0425d>`__
    if you are interested in details.

    .. image:: ../../img/nasnet_cell.png
       :width: 900
       :align: center

    Here is a glossary table, which could help better understand the terms used above:

    .. list-table::
        :widths: 25 75
        :header-rows: 1

        * - Name
          - Brief Description
        * - Cell
          - A cell consists of ``num_nodes`` nodes.
        * - Node
          - A node is the **sum** of ``num_ops_per_node`` operators.
        * - Operator
          - Each operator is independently chosen from a list of user-specified candidate operators.
        * - Operator's input
          - Each operator has one input, chosen from previous nodes as well as predecessors.
        * - Predecessors
          - Input of cell. A cell can have multiple predecessors. Predecessors are sent to *preprocessor* for preprocessing.
        * - Cell's output
          - Output of cell. Usually concatenation of some nodes (possibly all nodes) in the cell. Cell's output,
            along with predecessors, are sent to *postprocessor* for postprocessing.
        * - Preprocessor
          - Extra preprocessing to predecessors. Usually used in shape alignment (e.g., predecessors have different shapes).
            By default, do nothing.
        * - Postprocessor
          - Extra postprocessing for cell's output. Usually used to chain cells with multiple Predecessors
            (e.g., the next cell wants to have the outputs of both this cell and previous cell as its input).
            By default, directly use this cell's output.

    .. tip::

        It's highly recommended to make the candidate operators have an output of the same shape as input.
        This is because, there can be dynamic connections within cell. If there's shape change within operations,
        the input shape of the subsequent operation becomes unknown.
        In addition, the final concatenation could have shape mismatch issues.

    Parameters
    ----------
    op_candidates : list of module or function, or dict
        A list of modules to choose from, or a function that accepts current index and optionally its input index, and returns a module.
        For example, (2, 3, 0) means the 3rd op in the 2nd node, accepts the 0th node as input.
        The index are enumerated for all nodes including predecessors from 0.
        When first created, the input index is ``None``, meaning unknown.
        Note that in graph execution engine, support of function in ``op_candidates`` is limited.
        Please also note that, to make :class:`Cell` work with one-shot strategy,
        ``op_candidates``, in case it's a callable, should not depend on the second input argument,
        i.e., ``op_index`` in current node.
    num_nodes : int
        Number of nodes in the cell.
    num_ops_per_node: int
        Number of operators in each node. The output of each node is the sum of all operators in the node. Default: 1.
    num_predecessors : int
        Number of inputs of the cell. The input to forward should be a list of tensors. Default: 1.
    merge_op : "all", or "loose_end"
        If "all", all the nodes (except predecessors) will be concatenated as the cell's output, in which case, ``output_node_indices``
        will be ``list(range(num_predecessors, num_predecessors + num_nodes))``.
        If "loose_end", only the nodes that have never been used as other nodes' inputs will be concatenated to the output.
        Predecessors are not considered when calculating unused nodes.
        Details can be found in `NDS paper <https://arxiv.org/abs/1905.13214>`__. Default: all.
    preprocessor : callable
        Override this if some extra transformation on cell's input is intended.
        It should be a callable (``nn.Module`` is also acceptable) that takes a list of tensors which are predecessors,
        and outputs a list of tensors, with the same length as input.
        By default, it does nothing to the input.
    postprocessor : callable
        Override this if customization on the output of the cell is intended.
        It should be a callable that takes the output of this cell, and a list which are predecessors.
        Its return type should be either one tensor, or a tuple of tensors.
        The return value of postprocessor is the return value of the cell's forward.
        By default, it returns only the output of the current cell.
    concat_dim : int
        The result will be a concatenation of several nodes on this dim. Default: 1.
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    Examples
    --------
    Choose between conv2d and maxpool2d.
    The cell have 4 nodes, 1 op per node, and 2 predecessors.

    >>> cell = nn.Cell([nn.Conv2d(32, 32, 3, padding=1), nn.MaxPool2d(3, padding=1)], 4, 1, 2)

    In forward:

    >>> cell([input1, input2])

    The "list bracket" can be omitted:

    >>> cell(only_input)                    # only one input
    >>> cell(tensor1, tensor2, tensor3)     # multiple inputs

    Use ``merge_op`` to specify how to construct the output.
    The output will then have dynamic shape, depending on which input has been used in the cell.

    >>> cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2, merge_op='loose_end')
    >>> cell_out_channels = len(cell.output_node_indices) * 32

    The op candidates can be callable that accepts node index in cell, op index in node, and input index.

    >>> cell = nn.Cell([
    ...     lambda node_index, op_index, input_index: nn.Conv2d(32, 32, 3, stride=2 if input_index < 1 else 1),
    ... ], 4, 1, 2)

    Predecessor example: ::

        class Preprocessor:
            def __init__(self):
                self.conv1 = nn.Conv2d(16, 32, 1)
                self.conv2 = nn.Conv2d(64, 32, 1)

            def forward(self, x):
                return [self.conv1(x[0]), self.conv2(x[1])]

        cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2, preprocessor=Preprocessor())
        cell([torch.randn(1, 16, 48, 48), torch.randn(1, 64, 48, 48)])  # the two inputs will be sent to conv1 and conv2 respectively

    Warnings
    --------
    :class:`Cell` is not supported in :class:`~nni.nas.space.GraphModelSpace` model format.

    Attributes
    ----------
    output_node_indices : list of int
        An attribute that contains indices of the nodes concatenated to the output (a list of integers).

        When the cell is first instantiated in the base model, or when ``merge_op`` is ``all``,
        ``output_node_indices`` must be ``range(num_predecessors, num_predecessors + num_nodes)``.

        When ``merge_op`` is ``loose_end``, ``output_node_indices`` is useful to compute the shape of this cell's output,
        because the output shape depends on the connection in the cell, and which nodes are "loose ends" depends on mutation.

    op_candidates_factory : CellOpFactory or None
        If the operations are created with a factory (callable), this is to be set with the factory.
        One-shot algorithms will use this to make each node a cartesian product of operations and inputs.
    """

    def __init__(self,
                 op_candidates: Union[
                     Callable[[], List[nn.Module]],
                     List[nn.Module],
                     List[CellOpFactory],
                     Dict[str, nn.Module],
                     Dict[str, CellOpFactory]
                 ],
                 num_nodes: int,
                 num_ops_per_node: int = 1,
                 num_predecessors: int = 1,
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 preprocessor: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 postprocessor: Optional[Callable[[torch.Tensor, List[torch.Tensor]],
                                         Union[Tuple[torch.Tensor, ...], torch.Tensor]]] = None,
                 concat_dim: int = 1,
                 *,
                 label: Optional[str] = None):
        super().__init__()

        self.label_scope = label_scope(label)

        # modules are created in "natural" order
        # first create preprocessor
        self.preprocessor = preprocessor or _ListIdentity()
        # then create intermediate ops
        self.ops = nn.ModuleList()
        self.inputs = nn.ModuleList()
        # finally postprocessor
        self.postprocessor = postprocessor or _DefaultPostprocessor()

        self.num_nodes = num_nodes
        self.num_ops_per_node = num_ops_per_node
        self.num_predecessors = num_predecessors
        assert merge_op in ['all', 'loose_end']
        self.merge_op: Literal['all', 'loose_end'] = merge_op
        self.output_node_indices = list(range(num_predecessors, num_predecessors + num_nodes))

        self.concat_dim = concat_dim

        self.op_candidates_factory: Union[List[CellOpFactory], Dict[str, CellOpFactory], None] = None  # set later

        # fill-in the missing modules
        with self.label_scope:
            self._create_modules(op_candidates)

    def _create_modules(self, op_candidates):
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            self.ops.append(nn.ModuleList())
            self.inputs.append(nn.ModuleList())
            for k in range(self.num_ops_per_node):
                inp = InputChoice(i, 1, label=f'input_{i}_{k}')
                chosen = None

                if isinstance(inp, ChosenInputs):
                    # now we are in the fixed mode
                    # the length of chosen should be 1
                    chosen = inp.chosen[0]
                    if self.merge_op == 'loose_end' and chosen in self.output_node_indices:
                        # remove it from concat indices
                        self.output_node_indices.remove(chosen)

                # this is needed because op_candidates can be very complex
                # the type annoation and docs for details
                ops, has_factory = create_cell_op_candidates(op_candidates, i, k, chosen)
                if has_factory:
                    self.op_candidates_factory = op_candidates

                # though it's layer choice and input choice here, in fixed mode, the chosen module will be created.
                cast(nn.ModuleList, self.ops[-1]).append(LayerChoice(ops, label=f'op_{i}_{k}'))
                cast(nn.ModuleList, self.inputs[-1]).append(inp)

    def freeze(self, sample: Sample) -> 'Cell':
        if self.op_candidates_factory is not None:
            # Re-instantiate the whole cell if factory is used to create the operators.
            # The operators might be different when the inputs get fixed.

            _logger.debug('Recreating the cell `%s` to freeze as op factory is used.', self.label)
            with model_context(sample):
                return Cell(
                    self.op_candidates_factory,
                    self.num_nodes,
                    self.num_ops_per_node,
                    self.num_predecessors,
                    self.merge_op,
                    self.preprocessor,
                    self.postprocessor,
                    self.concat_dim,
                    label=self.label,
                )

        else:
            new_cell = cast(Cell, super().freeze(sample))

            # Only need to re-calculate the loose end indices
            if new_cell.merge_op == 'loose_end':
                used_nodes = set()
                for input_list in new_cell.inputs:
                    for input in input_list:  # type: ignore  # pylint: disable=redefined-builtin
                        assert isinstance(input, ChosenInputs)
                        used_nodes.update(input.chosen)

                new_cell.output_node_indices = [n for n in new_cell.output_node_indices if n not in used_nodes]

            return new_cell

    @property
    def label(self):
        return self.label_scope.name

    def forward(self, *inputs: Union[List[torch.Tensor], torch.Tensor]) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Forward propagation of cell.

        Parameters
        ----------
        inputs
            Can be a list of tensors, or several tensors.
            The length should be equal to ``num_predecessors``.

        Returns
        -------
        Tuple[torch.Tensor] | torch.Tensor
            The return type depends on the output of ``postprocessor``.
            By default, it's the output of ``merge_op``, which is a contenation (on ``concat_dim``)
            of some of (possibly all) the nodes' outputs in the cell.
        """
        processed_inputs: List[torch.Tensor] = preprocess_cell_inputs(self.num_predecessors, *inputs)
        states: List[torch.Tensor] = self.preprocessor(processed_inputs)
        for ops, inps in zip(
            cast(Sequence[Sequence[LayerChoice]], self.ops),
            cast(Sequence[Sequence[InputChoice]], self.inputs)
        ):
            current_state = []
            for op, inp in zip(ops, inps):
                current_state.append(op(inp(states)))
            current_state = torch.sum(torch.stack(current_state), 0)
            states.append(current_state)
        this_cell = torch.cat([states[k] for k in self.output_node_indices], self.concat_dim)
        return self.postprocessor(this_cell, processed_inputs)
