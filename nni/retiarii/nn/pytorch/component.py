import copy
from typing import Callable, List, Union, Tuple, Optional

import torch
import torch.nn as nn

from .api import LayerChoice, InputChoice
from .nn import ModuleList

from .utils import generate_new_label, get_fixed_value


__all__ = ['Repeat', 'Cell']


class Repeat(nn.Module):
    """
    Repeat a block by a variable number of times.

    Parameters
    ----------
    blocks : function, list of function, module or list of module
        The block to be repeated. If not a list, it will be replicated into a list.
        If a list, it should be of length ``max_depth``, the modules will be instantiated in order and a prefix will be taken.
        If a function, it will be called to instantiate a module. Otherwise the module will be deep-copied.
    depth : int or tuple of int
        If one number, the block will be repeated by a fixed number of times. If a tuple, it should be (min, max),
        meaning that the block will be repeated at least `min` times and at most `max` times.
    """

    def __new__(cls, blocks: Union[Callable[[], nn.Module], List[Callable[[], nn.Module]], nn.Module, List[nn.Module]],
                depth: Union[int, Tuple[int, int]], label: Optional[str] = None):
        try:
            repeat = get_fixed_value(label)
            return nn.Sequential(*cls._replicate_and_instantiate(blocks, repeat))
        except AssertionError:
            return super().__new__(cls)

    def __init__(self,
                 blocks: Union[Callable[[], nn.Module], List[Callable[[], nn.Module]], nn.Module, List[nn.Module]],
                 depth: Union[int, Tuple[int, int]], label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)
        self.min_depth = depth if isinstance(depth, int) else depth[0]
        self.max_depth = depth if isinstance(depth, int) else depth[1]
        assert self.max_depth >= self.min_depth > 0
        self.blocks = nn.ModuleList(self._replicate_and_instantiate(blocks, self.max_depth))

    @property
    def label(self):
        return self._label

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    @staticmethod
    def _replicate_and_instantiate(blocks, repeat):
        if not isinstance(blocks, list):
            if isinstance(blocks, nn.Module):
                blocks = [blocks] + [copy.deepcopy(blocks) for _ in range(repeat - 1)]
            else:
                blocks = [blocks for _ in range(repeat)]
        assert len(blocks) > 0
        assert repeat <= len(blocks), f'Not enough blocks to be used. {repeat} expected, only found {len(blocks)}.'
        blocks = blocks[:repeat]
        if not isinstance(blocks[0], nn.Module):
            blocks = [b() for b in blocks]
        return blocks


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
    merge_op : str
        Currently only ``all`` is supported, which has slight difference with that described in reference. Default: all.
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    References
    ----------
    .. [zophnas] Barret Zoph, Quoc V. Le, "Neural Architecture Search with Reinforcement Learning". https://arxiv.org/abs/1611.01578
    .. [zophnasnet] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le,
        "Learning Transferable Architectures for Scalable Image Recognition". https://arxiv.org/abs/1707.07012
    """

    # TODO:
    # Support loose end concat (shape inference on the following cells)
    # How to dynamically create convolution with stride as the first node

    def __init__(self,
                 op_candidates: Union[Callable, List[nn.Module]],
                 num_nodes: int,
                 num_ops_per_node: int = 1,
                 num_predecessors: int = 1,
                 merge_op: str = 'all',
                 label: str = None):
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
                self.ops[-1].append(LayerChoice(ops, label=f'{self.label}__op_{i}_{k}'))
                self.inputs[-1].append(InputChoice(i + num_predecessors, 1, label=f'{self.label}/input_{i}_{k}'))
        assert merge_op in ['all']  # TODO: loose_end
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
