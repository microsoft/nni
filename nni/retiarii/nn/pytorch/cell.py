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
from .nn import ModuleList
from .utils import generate_new_label


class Cell(nn.Module):
    """
    Cell structure [zophnas]_ [zophnasnet]_ that is popularly used in NAS literature.

    A cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from
    ``op_candidates``, and takes one input from previous nodes and predecessors. Predecessor means the input of cell.
    The output of cell is the concatenation of some of the nodes in the cell (currently all the nodes).

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
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    Attributes
    ----------
    output_node_indices : list of int
        Indices of the nodes concatenated to the output. For example, if the following operation is a 2d-convolution,
        its input channels is ``len(output_node_indices) * channels``.

    Examples
    --------
    >>> cell = nn.Cell([nn.Conv2d(32, 32, 3), nn.MaxPool2d(3)], 4, 1, 2)
    >>> output = cell([input1, input2])

    References
    ----------
    .. [zophnas] Barret Zoph, Quoc V. Le, "Neural Architecture Search with Reinforcement Learning". https://arxiv.org/abs/1611.01578
    .. [zophnasnet] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le,
        "Learning Transferable Architectures for Scalable Image Recognition". https://arxiv.org/abs/1707.07012
    .. [nds] Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Dollar, Piotr,
        "On Network Design Spaces for Visual Recognition". https://arxiv.org/abs/1905.13214
    """

    def __init__(self,
                 op_candidates: Union[
                     Callable[[], List[nn.Module]],
                     List[Union[nn.Module, Callable[[int, int, Optional[int]], nn.Module]]],
                     Dict[str, Union[nn.Module, Callable[[int, int, Optional[int]], nn.Module]]]
                 ],
                 num_nodes: int,
                 num_ops_per_node: int = 1,
                 num_predecessors: int = 1,
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 preprocessor: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 postprocessor: Optional[Callable[[torch.Tensor, List[torch.Tensor]],
                                         Union[Tuple[torch.Tensor], torch.Tensor]]] = None,
                 *,
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)

        # modules are created in "natural" order
        # first create preprocessor
        self.preprocessor = preprocessor or nn.Identity()
        # then create intermediate ops
        self.ops = ModuleList()
        self.inputs = ModuleList()
        # finally postprocessor
        self.postprocessor = postprocessor or (lambda this, prev: this)

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
                input = InputChoice(i, 1, label=f'{self.label}/input_{i}_{k}')
                chosen = None

                if isinstance(input, ChosenInputs):
                    # now we are in the fixed mode
                    # the length of chosen should be 1
                    chosen = input.chosen[0]
                    if self.merge_op == 'loose_end' and chosen in self.output_node_indices:
                        # remove it from concat indices
                        self.output_node_indices.remove(chosen)

                # this is needed because op_candidates can be very complex
                # the type annoation and docs for details
                ops = self._convert_op_candidates(op_candidates, i, k, chosen)

                # though it's layer choice and input choice here, in fixed mode, the chosen module will be created.
                self.ops[-1].append(LayerChoice(ops, label=f'{self.label}/op_{i}_{k}'))
                self.inputs[-1].append(input)

    @property
    def label(self):
        return self._label

    def forward(self, x: List[torch.Tensor]) -> Union[Tuple[torch.Tensor], torch.Tensor]:
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
