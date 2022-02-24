import copy
from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional

import torch
import torch.nn as nn

from .api import LayerChoice
from .cell import Cell
from .nasbench101 import NasBench101Cell, NasBench101Mutator
from .utils import Mutable, generate_new_label, get_fixed_value


__all__ = ['Repeat', 'Cell', 'NasBench101Cell', 'NasBench101Mutator', 'NasBench201Cell']


class Repeat(Mutable):
    """
    Repeat a block by a variable number of times.

    Parameters
    ----------
    blocks : function, list of function, module or list of module
        The block to be repeated. If not a list, it will be replicated (**deep-copied**) into a list.
        If a list, it should be of length ``max_depth``, the modules will be instantiated in order and a prefix will be taken.
        If a function, it will be called (the argument is the index) to instantiate a module.
        Otherwise the module will be deep-copied.
    depth : int or tuple of int
        If one number, the block will be repeated by a fixed number of times. If a tuple, it should be (min, max),
        meaning that the block will be repeated at least ``min`` times and at most ``max`` times.


    Examples
    --------
    Block() will be deep copied and repeated 3 times. ::

        self.blocks = nn.Repeat(Block(), 3)

    Block() will be repeated 1, 2, or 3 times. ::

        self.blocks = nn.Repeat(Block(), (1, 3))

    Can be used together with layer choice.
    With deep copy, the 3 layers will have the same label, thus share the choice. ::

        self.blocks = nn.Repeat(nn.LayerChoice([...]), (1, 3))

    To make the three layer choices independent,
    we need a factory function that accepts index (0, 1, 2, ...) and returns the module of the ``index``-th layer. ::

        self.blocks = nn.Repeat(lambda index: nn.LayerChoice([...], label=f'layer{index}'), (1, 3))
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
        """
        The forward of input choice is simply selecting first on all choices.
        It shouldn't be called directly by users in most cases.
        """
        tensors = [inputs]
        for layer in self.layers:
            current_tensor = []
            for i, op in enumerate(layer):
                current_tensor.append(op(tensors[i]))
            current_tensor = torch.sum(torch.stack(current_tensor), 0)
            tensors.append(current_tensor)
        return tensors[-1]
