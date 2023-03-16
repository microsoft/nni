# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import warnings
from contextlib import contextmanager
from typing import Callable, List, Union, Tuple, Optional, cast

import torch
import torch.nn as nn

from nni.mutable import Categorical, LabeledMutable, Mutable, Sample, SampleValidationError, ensure_frozen
from nni.mutable.mutable import MutableExpression
from nni.mutable.symbol import SymbolicExpression

from .base import MutableModule, recursive_freeze


__all__ = ['Repeat']

_logger = logging.getLogger(__name__)


class Repeat(MutableModule):
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

    @staticmethod
    def _canonicalize_depth(depth: Union[int, Tuple[int, int], Mutable], label: Optional[str]) -> Union[Categorical, int]:
        if isinstance(depth, tuple):
            low, high = depth
            assert 0 <= low <= high, f'Invalid range: [{low}, {high}]'
            return Categorical(list(range(low, high + 1)), label=label)
        elif isinstance(depth, (int, Mutable)):
            return cast(Union[Categorical, int], depth)
        else:
            raise TypeError(f'Unsupported depth type: {type(depth)}')

    @classmethod
    def create_fixed_module(cls,
                            current_model: dict,
                            blocks: Union[Callable[[int], nn.Module],
                                          List[Callable[[int], nn.Module]],
                                          nn.Module,
                                          List[nn.Module]],
                            depth: Union[int, Tuple[int, int], Mutable], *, label: Optional[str] = None):
        depth = cls._canonicalize_depth(depth, label)
        if isinstance(depth, Mutable):
            depth = depth.freeze(current_model)
        # It can be a int initially, or frozen to be int just now.
        if not isinstance(depth, int):
            raise TypeError(f'depth must be frozen to int when arch is not None: {depth}')
        return nn.Sequential(*cls._replicate_and_instantiate(blocks, depth))

    def __init__(self,
                 blocks: Union[Callable[[int], nn.Module],
                               List[Callable[[int], nn.Module]],
                               nn.Module,
                               List[nn.Module]],
                 depth: Union[int, Tuple[int, int], SymbolicExpression],
                 *,
                 label: Optional[str] = None):
        super().__init__()

        self._label = None  # by default, no label

        if isinstance(depth, SymbolicExpression):
            assert isinstance(depth, Mutable), 'depth must be Mutable and SymbolicExpression at the same time.'
            if label is not None:
                warnings.warn(
                    'In repeat, `depth` is already a mutable, but `label` is still set. It will be ignored.',
                    RuntimeWarning
                )

            self.depth_choice: Union[int, MutableExpression] = cast(Union[int, MutableExpression], depth)
            all_values = _all_finite_integers(depth)
            self.min_depth = min(all_values)
            self.max_depth = max(all_values)

        elif isinstance(depth, tuple):
            self.min_depth = depth if isinstance(depth, int) else depth[0]
            self.max_depth = depth if isinstance(depth, int) else depth[1]
            self.depth_choice: Union[int, MutableExpression] = Categorical(list(range(self.min_depth, self.max_depth + 1)), label=label)
            self._label = self.depth_choice.label

        elif isinstance(depth, int):
            self.min_depth = self.max_depth = depth
            self.depth_choice: Union[int, MutableExpression] = depth
        else:
            raise TypeError(f'Unsupported "depth" type: {type(depth)}')
        assert self.max_depth >= self.min_depth >= 0 and self.max_depth >= 1, f'Depth of {self.min_depth} to {self.max_depth} is invalid.'

        if isinstance(self.depth_choice, Mutable):
            self.add_mutable(self.depth_choice)
            self._dry_run_depth = ensure_frozen(self.depth_choice)
        else:
            self._dry_run_depth = self.depth_choice

        if isinstance(blocks, nn.ModuleList):
            _logger.warning('Using ModuleList as blocks will make the whole module list be treated as one block, '
                            'and it will be deep-copied. This might be not intended in most cases. '
                            'Consider using a pure python list of modules if you want to specify a sequence of blocks.')

        self.blocks = nn.ModuleList(self._replicate_and_instantiate(blocks, self.max_depth))

    @torch.jit.unused
    @property
    def label(self) -> Optional[str]:
        if isinstance(self.depth_choice, LabeledMutable):
            return self.depth_choice.label
        return None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i < self._dry_run_depth:
                x = block(x)
            # Make JIT happy
            # else:
            #     break
        return x

    def check_contains(self, sample: Sample) -> Optional[SampleValidationError]:
        # Check depth choice
        if isinstance(self.depth_choice, Mutable):
            exception = self.depth_choice.check_contains(sample)
            if exception is not None:
                return exception
            depth = self.depth_choice.freeze(sample)
        else:
            depth = self.depth_choice

        # Check blocks
        for i, block in enumerate(self.blocks):
            if i < depth:
                exception = self._check_any_module_contains(block, sample, str(i))
                if exception is not None:
                    return exception

        return None

    @staticmethod
    def _check_any_module_contains(module: nn.Module, sample: Sample, path: str) -> Optional[SampleValidationError]:
        if isinstance(module, MutableModule):
            exception = module.check_contains(sample)
            if exception is not None:
                exception.paths.append(path)
                return exception
        else:
            for name, module in MutableModule.named_mutable_descendants(module):  # type: ignore
                exception = module.check_contains(sample)
                if exception is not None:
                    exception.paths.append(name)
                    exception.paths.append(path)
                    return exception

        return None

    def freeze(self, sample: Sample) -> nn.Sequential:
        self.validate(sample)
        if isinstance(self.depth_choice, Mutable):
            depth = self.depth_choice.freeze(sample)
            assert isinstance(depth, int), 'Depth must be frozen to int.'
        elif isinstance(self.depth_choice, int):
            depth = self.depth_choice
        else:
            raise TypeError(f'Unsupported depth type: {type(self.depth_choice)}')
        blocks = []
        for i, block in enumerate(self.blocks):
            if i < depth:
                blocks.append(recursive_freeze(block, sample)[0])
            # Make JIT happy
            # else:
            #     break
        return nn.Sequential(*blocks)

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
        # TODO: The blocks can't contain auto-generated label. Otherwise it won't match. We need to detect this.
        return blocks

    def __getitem__(self, index):
        # shortcut for blocks[index]
        return self.blocks[index]

    def __len__(self):
        return self.max_depth


@contextmanager
def repeat_jit_forward_patch():
    """
    Patch the forward method of Repeat to make it JIT friendly.
    Using ``if`` in forward will cause the graph to be nasty and hard to mutate.
    """

    def new_forward(self: Repeat, x):
        for block in self.blocks:
            x = block(x)
        return x

    old_forward = Repeat.forward
    try:
        Repeat.forward = new_forward
        yield
    finally:
        Repeat.forward = old_forward


def _all_finite_integers(mutable: Mutable) -> List[int]:
    all_values = list(mutable.grid())
    all_values_fine_grained = list(mutable.grid(granularity=2))
    if all_values != all_values_fine_grained:
        raise ValueError(
            f'Invalid depth choice: {mutable}. '
            f'Only support discrete values, but some variables might be continuous.'
        )
    if not all(isinstance(val, int) for val in all_values):
        raise ValueError(
            f'Invalid depth choice: {mutable}. '
            f'Only choice of integers. But some choices are not: {all_values}'
        )
    return all_values
