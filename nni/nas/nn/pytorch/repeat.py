# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import warnings
from typing import Callable, List, Union, Tuple, Optional

import torch.nn as nn

from nni.nas.utils import NoContextError, STATE_DICT_PY_MAPPING_PARTIAL

from .choice import ValueChoice, ValueChoiceX, ChoiceOf
from .mutation_utils import Mutable, get_fixed_value


__all__ = ['Repeat']


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

        .. versionadded:: 2.8

           Minimum depth can be 0. But this feature is NOT supported on graph engine.

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
                            depth: Union[int, Tuple[int, int], ChoiceOf[int]], *, label: Optional[str] = None):
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
                 depth: Union[int, Tuple[int, int], ChoiceOf[int]], *, label: Optional[str] = None):
        super().__init__()

        self._label = None  # by default, no label

        if isinstance(depth, ValueChoiceX):
            if label is not None:
                warnings.warn(
                    'In repeat, `depth` is already a ValueChoice, but `label` is still set. It will be ignored.',
                    RuntimeWarning
                )
            self.depth_choice: Union[int, ChoiceOf[int]] = depth
            all_values = list(self.depth_choice.all_options())
            self.min_depth = min(all_values)
            self.max_depth = max(all_values)

            if isinstance(depth, ValueChoice):
                self._label = depth.label  # if a leaf node

        elif isinstance(depth, tuple):
            self.min_depth = depth if isinstance(depth, int) else depth[0]
            self.max_depth = depth if isinstance(depth, int) else depth[1]
            self.depth_choice: Union[int, ChoiceOf[int]] = ValueChoice(list(range(self.min_depth, self.max_depth + 1)), label=label)
            self._label = self.depth_choice.label

        elif isinstance(depth, int):
            self.min_depth = self.max_depth = depth
            self.depth_choice: Union[int, ChoiceOf[int]] = depth
        else:
            raise TypeError(f'Unsupported "depth" type: {type(depth)}')
        assert self.max_depth >= self.min_depth >= 0 and self.max_depth >= 1, f'Depth of {self.min_depth} to {self.max_depth} is invalid.'
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
                blocks = [blocks if i == 0 else copy.deepcopy(blocks) for i in range(repeat)]
            else:
                blocks = [blocks for _ in range(repeat)]
        assert repeat <= len(blocks), f'Not enough blocks to be used. {repeat} expected, only found {len(blocks)}.'
        if repeat < len(blocks):
            blocks = blocks[:repeat]
        if len(blocks) > 0 and not isinstance(blocks[0], nn.Module):
            blocks = [b(i) for i, b in enumerate(blocks)]
        return blocks

    def __getitem__(self, index):
        # shortcut for blocks[index]
        return self.blocks[index]

    def __len__(self):
        return self.max_depth
