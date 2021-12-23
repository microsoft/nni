import copy
from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional

import torch
import torch.nn as nn

from .api import LayerChoice, InputChoice
from .nn import ModuleList

from .nasbench101 import NasBench101Cell, NasBench101Mutator
from .utils import Mutable, generate_new_label, get_fixed_value


__all__ = ['Repeat', 'Cell', 'NasBench101Cell', 'NasBench101Mutator', 'NasBench201Cell']


class Repeat(Mutable):
    """
    Repeat a block by a variable number of times.

    Parameters
    ----------
    blocks : function, list of function, module or list of module
        The block to be repeated. If not a list, it will be replicated into a list.
        If a list, it should be of length ``max_depth``, the modules will be instantiated in order and a prefix will be taken.
        If a function, it will be called (the argument is the index) to instantiate a module.
        Otherwise the module will be deep-copied.
    depth : int or tuple of int
        If one number, the block will be repeated by a fixed number of times. If a tuple, it should be (min, max),
        meaning that the block will be repeated at least `min` times and at most `max` times.
    """

    @classmethod
    def create_fixed_module(cls,
                            blocks: Union[Callable[[int], nn.Module],
                                          List[Callable[[int], nn.Module]],
                                          nn.Module,
                                          List[nn.Module]],
                            depth: Union[int, Tuple[int, int]], *, label: Optional[str] = None):
        repeat = get_fixed_value(label)
        return nn.Sequential(*cls._replicate_and_instantiate(blocks, repeat))

    def __init__(self,
                 blocks: Union[Callable[[int], nn.Module],
                               List[Callable[[int], nn.Module]],
                               nn.Module,
                               List[nn.Module]],
                 depth: Union[int, Tuple[int, int]], *, label: Optional[str] = None):
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
            blocks = [b(i) for i, b in enumerate(blocks)]
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


class NasBench201Cell(nn.Module):
    """
    Cell structure that is proposed in NAS-Bench-201 [nasbench201]_ .

    This cell is a densely connected DAG with ``num_tensors`` nodes, where each node is tensor.
    For every i < j, there is an edge from i-th node to j-th node.
    Each edge in this DAG is associated with an operation transforming the hidden state from the source node
    to the target node. All possible operations are selected from a predefined operation set, defined in ``op_candidates``.
    Each of the ``op_candidates`` should be a callable that accepts input dimension and output dimension,
    and returns a ``Module``.

    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be :math:`[N, C_{out}, *]`. For example,

    The space size of this cell would be :math:`|op|^{N(N-1)/2}`, where :math:`|op|` is the number of operation candidates,
    and :math:`N` is defined by ``num_tensors``.

    Parameters
    ----------
    op_candidates : list of callable
        Operation candidates. Each should be a function accepts input feature and output feature, returning nn.Module.
    in_features : int
        Input dimension of cell.
    out_features : int
        Output dimension of cell.
    num_tensors : int
        Number of tensors in the cell (input included). Default: 4
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.

    References
    ----------
    .. [nasbench201] Dong, X. and Yang, Y., 2020. Nas-bench-201: Extending the scope of reproducible neural architecture search.
        arXiv preprint arXiv:2001.00326.
    """

    @staticmethod
    def _make_dict(x):
        if isinstance(x, list):
            return OrderedDict([(str(i), t) for i, t in enumerate(x)])
        return OrderedDict(x)

    def __init__(self, op_candidates: List[Callable[[int, int], nn.Module]],
                 in_features: int, out_features: int, num_tensors: int = 4,
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)

        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tensors = num_tensors

        op_candidates = self._make_dict(op_candidates)

        for tid in range(1, num_tensors):
            node_ops = nn.ModuleList()
            for j in range(tid):
                inp = in_features if j == 0 else out_features
                op_choices = OrderedDict([(key, cls(inp, out_features))
                                          for key, cls in op_candidates.items()])
                node_ops.append(LayerChoice(op_choices, label=f'{self._label}__{j}_{tid}'))  # put __ here to be compatible with base engine
            self.layers.append(node_ops)

    def forward(self, inputs):
        tensors = [inputs]
        for layer in self.layers:
            current_tensor = []
            for i, op in enumerate(layer):
                current_tensor.append(op(tensors[i]))
            current_tensor = torch.sum(torch.stack(current_tensor), 0)
            tensors.append(current_tensor)
        return tensors[-1]
