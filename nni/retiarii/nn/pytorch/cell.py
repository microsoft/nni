import copy
import warnings
from typing import Callable, Dict, List, Union, Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn

from .api import ChosenInputs, LayerChoice, InputChoice
from .nn import ModuleList  # pylint: disable=no-name-in-module
from .mutation_utils import generate_new_label


class _ListIdentity(nn.Identity):
    # workaround for torchscript
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return x


class _DefaultPostprocessor(nn.Module):
    # this is also a workaround for torchscript

    def forward(self, this_cell: torch.Tensor, prev_cell: List[torch.Tensor]) -> torch.Tensor:
        return this_cell


_cell_op_factory_type = Callable[[int, int, Optional[int]], nn.Module]


class Cell(nn.Module):
    """
    Cell structure that is popularly used in NAS literature.

    Refer to :footcite:t:`zoph2017neural,zoph2018learning,liu2018darts` for details.
    :footcite:t:`radosavovic2019network` is a good summary of how this structure works in practice.

    A cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from
    ``op_candidates``, and takes one input from previous nodes and predecessors. Predecessor means the input of cell.
    The output of cell is the concatenation of some of the nodes in the cell (currently all the nodes).

    Here is a glossary table, which could help better understand the terms used above:

    .. list-table::
        :widths: 25 75

        * - Cell
          - A cell consists of several nodes.
        * - Node
          - A node is the **sum** of several operators.
        * - Operator
          - Each operator is independently chosen from a list of user-specified candidate operators.
        * - Operator's input
          - Each operator has one input, chosen from previous nodes as well as predecessors.
        * - Predecessors
          - Input of cell. A cell can have multiple predecessors. Predecessors are sent to *preprocessor* for preprocessing.
        * - Cell's output
          - Output of cell. Usually concatenation of several nodes (possibly all nodes) in the cell. Cell's output,
            along with predecessors, are sent to *postprocessor* for postprocessing.
        * - Preprocessor
          - Extra preprocessing to predecessors. Usually used in shape alignment (e.g., predecessors have different shapes).
            By default, do nothing.
        * - Postprocessor
          - Extra postprocessing for cell's output. Usually used to chain cells with multiple Predecessors
            (e.g., the next cell wants to have the outputs of both this cell and previous cell as its input).
            By default, directly use this cell's output.

    Parameters
    ----------
    op_candidates : list of module or function, or dict
        A list of modules to choose from, or a function that accepts current index and optionally its input index, and returns a module.
        For example, (2, 3, 0) means the 3rd op in the 2nd node, accepts the 0th node as input.
        The index are enumerated for all nodes including predecessors from 0.
        When first created, the input index is ``None``, meaning unknown.
        Note that in graph execution engine, support of function in ``op_candidates`` is limited.
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
        Details can be found in reference [nds]. Default: all.
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
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    Attributes
    ----------
    output_node_indices : list of int
        Indices of the nodes concatenated to the output. For example, if the following operation is a 2d-convolution,
        its input channels is ``len(output_node_indices) * channels``.

    Examples
    --------
    Choose between conv2d and maxpool2d.
    The cell have 4 nodes, 1 op per node, and 2 predecessors.
    >>> cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2)
    In forward:
    >>> cell([input1, input2])

    Use ``merge_op`` to specify how to construct the output.
    The output will then have dynamic shape, depending on which input has been used in the cell.
    >>> cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2, merge_op='loose_end')

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
    """

    def __init__(self,
                 op_candidates: Union[
                     Callable[[], List[nn.Module]],
                     List[Union[nn.Module, _cell_op_factory_type]],
                     Dict[str, Union[nn.Module, _cell_op_factory_type]]
                 ],
                 num_nodes: int,
                 num_ops_per_node: int = 1,
                 num_predecessors: int = 1,
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 preprocessor: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 postprocessor: Optional[Callable[[torch.Tensor, List[torch.Tensor]],
                                         Union[Tuple[torch.Tensor, ...], torch.Tensor]]] = None,
                 *,
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)

        # modules are created in "natural" order
        # first create preprocessor
        self.preprocessor = preprocessor or _ListIdentity()
        # then create intermediate ops
        self.ops = ModuleList()
        self.inputs = ModuleList()
        # finally postprocessor
        self.postprocessor = postprocessor or _DefaultPostprocessor()

        self.num_nodes = num_nodes
        self.num_ops_per_node = num_ops_per_node
        self.num_predecessors = num_predecessors
        assert merge_op in ['all', 'loose_end']
        self.merge_op = merge_op
        self.output_node_indices = list(range(num_predecessors, num_predecessors + num_nodes))

        # fill-in the missing modules
        self._create_modules(op_candidates)

    def _create_modules(self, op_candidates):
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            self.ops.append(ModuleList())
            self.inputs.append(ModuleList())
            for k in range(self.num_ops_per_node):
                inp = InputChoice(i, 1, label=f'{self.label}/input_{i}_{k}')
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
                ops = self._convert_op_candidates(op_candidates, i, k, chosen)

                # though it's layer choice and input choice here, in fixed mode, the chosen module will be created.
                self.ops[-1].append(LayerChoice(ops, label=f'{self.label}/op_{i}_{k}'))
                self.inputs[-1].append(inp)

    @property
    def label(self):
        return self._label

    def forward(self, x: List[torch.Tensor]):
        # The return type should be 'Union[Tuple[torch.Tensor, ...], torch.Tensor]'.
        # Cannot decorate it as annotation. Otherwise torchscript will complain.
        assert isinstance(x, list), 'We currently only support input of cell as a list, even if you have only one predecessor.'
        states = self.preprocessor(x)
        for ops, inps in zip(self.ops, self.inputs):
            current_state = []
            for op, inp in zip(ops, inps):
                current_state.append(op(inp(states)))
            current_state = torch.sum(torch.stack(current_state), 0)
            states.append(current_state)
        if self.merge_op == 'all':
            # a special case for graph engine
            this_cell = torch.cat(states[self.num_predecessors:], 1)
        else:
            this_cell = torch.cat([states[k] for k in self.output_node_indices], 1)
        return self.postprocessor(this_cell, x)

    @staticmethod
    def _convert_op_candidates(op_candidates, node_index, op_index, chosen) -> Union[Dict[str, nn.Module], List[nn.Module]]:
        # convert the complex type into the type that is acceptable to LayerChoice
        def convert_single_op(op):
            if isinstance(op, nn.Module):
                return copy.deepcopy(op)
            elif callable(op):
                # FIXME: I don't know how to check whether we are in graph engine.
                return op(node_index, op_index, chosen)
            else:
                raise TypeError(f'Unrecognized type {type(op)} for op {op}')

        if isinstance(op_candidates, list):
            return [convert_single_op(op) for op in op_candidates]
        elif isinstance(op_candidates, dict):
            return {key: convert_single_op(op) for key, op in op_candidates.items()}
        elif callable(op_candidates):
            warnings.warn(f'Directly passing a callable into Cell is deprecated. Please consider migrating to list or dict.',
                          DeprecationWarning)
            return op_candidates()
        else:
            raise TypeError(f'Unrecognized type {type(op_candidates)} for {op_candidates}')
