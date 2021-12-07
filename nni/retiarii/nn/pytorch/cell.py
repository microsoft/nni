import copy
from typing import Callable, List, Union, Tuple, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn

from nni.retiarii.utils import NoContextError
from .api import LayerChoice, InputChoice
from .nn import ModuleList
from .utils import generate_new_label, get_fixed_value, get_fixed_dict


class Cell(nn.Module):
    """
    Cell structure [zophnas]_ [zophnasnet]_ that is popularly used in NAS literature.

    A cell consists of multiple "nodes". Each node is a sum of multiple operators. Each operator is chosen from
    ``op_candidates``, and takes one input from previous nodes and predecessors. Predecessor means the input of cell.
    The output of cell is the concatenation of some of the nodes in the cell (currently all the nodes).

    Parameters
    ----------
    op_candidates : function or list of module
        A list of modules to choose from, or a function that returns a list of modules.
    num_nodes : int
        Number of nodes in the cell.
    num_ops_per_node: int
        Number of operators in each node. The output of each node is the sum of all operators in the node. Default: 1.
    num_predecessors : int
        Number of inputs of the cell. The input to forward should be a list of tensors. Default: 1.
    merge_op : "all", or "loose_end"
        If "all", all the nodes (except predecessors) will be concatenated as the cell's output.
        If "loose_end", only the nodes that have never been used as other nodes' inputs will be concatenated to the output.
        Details can be found in reference [nds]. Default: all.
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    References
    ----------
    .. [zophnas] Barret Zoph, Quoc V. Le, "Neural Architecture Search with Reinforcement Learning". https://arxiv.org/abs/1611.01578
    .. [zophnasnet] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le,
        "Learning Transferable Architectures for Scalable Image Recognition". https://arxiv.org/abs/1707.07012
    .. [nds] Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Dollar, Piotr,
        "On Network Design Spaces for Visual Recognition". https://arxiv.org/abs/1905.13214
    """

    # TODO:
    # How to dynamically create convolution with stride as the first node

    def __new__(cls,
                op_candidates: Union[Callable, List[nn.Module]],
                num_nodes: int,
                num_ops_per_node: int = 1,
                num_predecessors: int = 1,
                merge_op: Literal['all', 'loose_end'] = 'all',
                label: Optional[str] = None):
        def make_list(x): return x if isinstance(x, list) else [x]

        try:
            label, selected = get_fixed_dict(label)
            op_candidates = cls._make_dict(op_candidates)
            num_nodes = selected[f'{label}/num_nodes']
            adjacency_list = [make_list(selected[f'{label}/input{i}']) for i in range(1, num_nodes)]
            if sum([len(e) for e in adjacency_list]) > max_num_edges:
                raise InvalidMutation(f'Expected {max_num_edges} edges, found: {adjacency_list}')
            return _NasBench101CellFixed(
                [op_candidates[selected[f'{label}/op{i}']] for i in range(1, num_nodes - 1)],
                adjacency_list, in_features, out_features, num_nodes, projection)
        except NoContextError:
            return super().__new__(cls)

    def __init__(self,
                 op_candidates: Union[Callable, List[nn.Module]],
                 num_nodes: int,
                 num_ops_per_node: int = 1,
                 num_predecessors: int = 1,
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)
        self.ops = ModuleList()
        self.inputs = ModuleList()
        self.num_nodes = num_nodes
        self.num_ops_per_node = num_ops_per_node
        self.num_predecessors = num_predecessors
        for i in range(num_nodes):
            self.ops.append(ModuleList())
            self.inputs.append(ModuleList())
            for k in range(num_ops_per_node):
                if isinstance(op_candidates, list):
                    assert len(op_candidates) > 0 and isinstance(op_candidates[0], nn.Module)
                    ops = copy.deepcopy(op_candidates)
                else:
                    ops = op_candidates()
                self.ops[-1].append(LayerChoice(ops, label=f'{self.label}/op_{i}_{k}'))
                self.inputs[-1].append(InputChoice(i + num_predecessors, 1, label=f'{self.label}/input_{i}_{k}'))
        assert merge_op in ['all', 'str']
        self.merge_op = merge_op

    @property
    def label(self):
        return self._label

    def forward(self, x: List[torch.Tensor]):
        states = x
        for ops, inps in zip(self.ops, self.inputs):
            current_state = []
            for op, inp in zip(ops, inps):
                current_state.append(op(inp(states)))
            current_state = torch.sum(torch.stack(current_state), 0)
            states.append(current_state)
        return torch.cat(states[self.num_predecessors:], 1)
