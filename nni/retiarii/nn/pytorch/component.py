import copy
import warnings
from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional

import torch
import torch.nn as nn

from nni.retiarii.utils import NoContextError, STATE_DICT_PY_MAPPING_PARTIAL

from .api import LayerChoice, ValueChoice, ValueChoiceX
from .cell import Cell
from .nasbench101 import NasBench101Cell, NasBench101Mutator
from .mutation_utils import Mutable, generate_new_label, get_fixed_value


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
        If a ValueChoice, it should choose from a series of positive integers.

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

    Depth can be a ValueChoice to support arbitrary depth candidate list. ::

        self.blocks = nn.Repeat(Block(), nn.ValueChoice([1, 3, 5]))
    """

    @classmethod
    def create_fixed_module(cls,
                            blocks: Union[Callable[[int], nn.Module],
                                          List[Callable[[int], nn.Module]],
                                          nn.Module,
                                          List[nn.Module]],
                            depth: Union[int, Tuple[int, int], ValueChoice], *, label: Optional[str] = None):
        if isinstance(depth, tuple):
            # we can't create a value choice here,
            # otherwise we will have two value choices, one created here, another in init.
            depth = get_fixed_value(label)

        if isinstance(depth, int):
            # if depth is a valuechoice, it should be already an int
            result = nn.Sequential(*cls._replicate_and_instantiate(blocks, depth))

            if hasattr(result, STATE_DICT_PY_MAPPING_PARTIAL):
                # already has a mapping, will merge with it
                prev_mapping = getattr(result, STATE_DICT_PY_MAPPING_PARTIAL)
                setattr(result, STATE_DICT_PY_MAPPING_PARTIAL, {k: f'blocks.{v}' for k, v in prev_mapping.items()})
            else:
                setattr(result, STATE_DICT_PY_MAPPING_PARTIAL, {'__self__': 'blocks'})

            return result

        raise NoContextError(f'Not in fixed mode, or {depth} not an integer.')

    def __init__(self,
                 blocks: Union[Callable[[int], nn.Module],
                               List[Callable[[int], nn.Module]],
                               nn.Module,
                               List[nn.Module]],
                 depth: Union[int, Tuple[int, int]], *, label: Optional[str] = None):
        super().__init__()

        self._label = None  # by default, no label

        if isinstance(depth, ValueChoiceX):
            if label is not None:
                warnings.warn(
                    'In repeat, `depth` is already a ValueChoice, but `label` is still set. It will be ignored.',
                    RuntimeWarning
                )
            self.depth_choice = depth
            all_values = list(self.depth_choice.all_options())
            self.min_depth = min(all_values)
            self.max_depth = max(all_values)

            if isinstance(depth, ValueChoice):
                self._label = depth.label  # if a leaf node

        elif isinstance(depth, tuple):
            self.min_depth = depth if isinstance(depth, int) else depth[0]
            self.max_depth = depth if isinstance(depth, int) else depth[1]
            self.depth_choice = ValueChoice(list(range(self.min_depth, self.max_depth + 1)), label=label)
            self._label = self.depth_choice.label

        elif isinstance(depth, int):
            self.min_depth = self.max_depth = depth
            self.depth_choice = depth
        else:
            raise TypeError(f'Unsupported "depth" type: {type(depth)}')
        assert self.max_depth >= self.min_depth > 0
        self.blocks = nn.ModuleList(self._replicate_and_instantiate(blocks, self.max_depth))

    @property
    def label(self) -> Optional[str]:
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

    def __getitem__(self, index):
        # shortcut for blocks[index]
        return self.blocks[index]

    def __len__(self):
        return self.max_depth


class NasBench201Cell(nn.Module):
    """
    Cell structure that is proposed in NAS-Bench-201.

    Proposed by `NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search <https://arxiv.org/abs/2001.00326>`__.

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
