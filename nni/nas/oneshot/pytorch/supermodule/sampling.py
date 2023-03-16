# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# type: ignore

from __future__ import annotations

import copy
import functools
import random
from typing import Any, List, Dict, Sequence, cast

import torch
import torch.nn as nn

from nni.mutable import MutableExpression, label_scope, Mutable, Categorical, CategoricalMultiple
from nni.nas.nn.pytorch import LayerChoice, InputChoice, Repeat, Cell
from nni.nas.nn.pytorch.cell import CellOpFactory, create_cell_op_candidates, preprocess_cell_inputs

from .base import BaseSuperNetModule
from ._expression_utils import weighted_sum
from .operation import MixedOperationSamplingPolicy, MixedOperation

__all__ = [
    'PathSamplingLayer', 'PathSamplingInput',
    'PathSamplingRepeat', 'PathSamplingCell',
    'MixedOpPathSamplingPolicy'
]


class PathSamplingLayer(LayerChoice, BaseSuperNetModule):
    """
    Mixed layer, in which fprop is decided by exactly one inner layer or sum of multiple (sampled) layers.
    If multiple modules are selected, the result will be summed and returned.

    Attributes
    ----------
    _sampled : int or list of str
        Sampled module indices.
    label : str
        Name of the choice.
    """

    def __init__(self, paths: dict[str, nn.Module] | list[nn.Module], label: str):
        super().__init__(paths, label=label)
        self._sampled: list[str] | str | None = None  # sampled can be either a list of indices or an index

    def resample(self, memo):
        """Random choose one path if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            self._sampled = self.choice.random()
        return {self.label: self._sampled}

    def export(self, memo):
        """Random choose one name if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self.choice.random()}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) is LayerChoice:
            return cls(module.candidates, module.label)

    def _reduction(self, items: list[Any], sampled: list[Any]):
        """Override this to implement customized reduction."""
        return weighted_sum(items)

    def forward(self, *args, **kwargs):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        res = [self[samp](*args, **kwargs) for samp in sampled]
        return self._reduction(res, sampled)


class PathSamplingInput(InputChoice, BaseSuperNetModule):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled input indices.
    """

    def __init__(self, n_candidates: int, n_chosen: int, reduction: str, label: str):
        super().__init__(n_candidates, n_chosen=n_chosen, reduction=reduction, label=label)
        self._sampled: list[int] | int | None = None

    def resample(self, memo):
        """Random choose one path / multiple paths if label is not found in memo.
        If one path is selected, only one integer will be in ``self._sampled``.
        If multiple paths are selected, a list will be in ``self._sampled``.
        """
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            self._sampled = self.choice.random()
        return {self.label: self._sampled}

    def export(self, memo):
        """Random choose one name if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self.choice.random()}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) is InputChoice:
            if module.reduction not in ['sum', 'mean', 'concat']:
                raise ValueError('Only input choice of sum/mean/concat reduction is supported.')
            if module.n_chosen is None:
                raise ValueError('n_chosen is None is not supported yet.')
            return cls(module.n_candidates, module.n_chosen, module.reduction, module.label)

    def _reduction(self, items: list[Any], sampled: list[Any]) -> Any:
        """Override this to implement customized reduction."""
        if len(items) == 1:
            return items[0]
        else:
            if self.reduction == 'sum':
                return sum(items)
            elif self.reduction == 'mean':
                return sum(items) / len(items)
            elif self.reduction == 'concat':
                return torch.cat(items, 1)
            raise ValueError(f'Unsupported reduction type: {self.reduction}')

    def forward(self, input_tensors):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        if len(input_tensors) != self.n_candidates:
            raise ValueError(f'Expect {self.n_candidates} input tensors, found {len(input_tensors)}.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled
        res = [input_tensors[samp] for samp in sampled]
        return self._reduction(res, sampled)


class MixedOpPathSamplingPolicy(MixedOperationSamplingPolicy):
    """Implements the path sampling in mixed operation.

    One mixed operation can have multiple value choices in its arguments.
    Each value choice can be further decomposed into "leaf value choices".
    We sample the leaf nodes, and composits them into the values on arguments.
    """

    def __init__(self, operation: MixedOperation, memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> None:
        # Sampling arguments. This should have the same keys with `operation.mutable_arguments`
        self._sampled: dict[str, Any] | None = None

    def resample(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Random sample for each leaf value choice."""
        result = {}
        space_spec = operation.simplify()
        for label, mutable in space_spec.items():
            if label in memo:
                result[label] = memo[label]
            else:
                result[label] = mutable.random()

        # composites to kwargs
        # example: result = {"exp_ratio": 3}, self._sampled = {"in_channels": 48, "out_channels": 96}
        self._sampled = {}
        for key, value in operation.mutable_arguments.items():
            self._sampled[key] = value.freeze(result)

        return result

    def export(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Export is also random for each leaf value choice."""
        result = {}
        space_spec = operation.simplify()
        for label, mutable in space_spec.items():
            if label not in memo:
                result[label] = mutable.random()
        return result

    def forward_argument(self, operation: MixedOperation, name: str) -> Any:
        # NOTE: we don't support sampling a list here.
        if self._sampled is None:
            raise ValueError('Need to call resample() before running forward')
        if name in operation.mutable_arguments:
            return self._sampled[name]
        return operation.init_arguments[name]


class PathSamplingRepeat(Repeat, BaseSuperNetModule):
    """
    Implementation of Repeat in a path-sampling supernet.
    Samples one / some of the prefixes of the repeated blocks.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled depth.
    """

    def __init__(self, blocks: list[nn.Module], depth: MutableExpression[int]):
        super().__init__(blocks, depth)
        self._sampled: list[int] | int | None = None

    def resample(self, memo):
        """Since depth is based on ValueChoice, we only need to randomly sample every leaf value choices."""
        result = {}
        assert isinstance(self.depth_choice, Mutable)
        for label, mutable in self.depth_choice.simplify().items():
            if label in memo:
                result[label] = memo[label]
            else:
                result[label] = mutable.random()

        self._sampled = self.depth_choice.freeze(result)

        return result

    def export(self, memo):
        """Random choose one if every choice not in memo."""
        result = {}
        assert isinstance(self.depth_choice, Mutable)
        for label, mutable in self.depth_choice.simplify().items():
            if label not in memo:
                result[label] = mutable.random()
        return result

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) == Repeat and isinstance(module.depth_choice, MutableExpression):
            # Only interesting when depth is mutable
            return cls(list(module.blocks), module.depth_choice)

    def _reduction(self, items: list[Any], sampled: list[Any]):
        """Override this to implement customized reduction."""
        return weighted_sum(items)

    def forward(self, x):
        if self._sampled is None:
            raise RuntimeError('At least one depth needs to be sampled before fprop.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        res = []
        for cur_depth, block in enumerate(self.blocks, start=1):
            x = block(x)
            if cur_depth in sampled:
                res.append(x)
            if not any(d > cur_depth for d in sampled):
                break
        return self._reduction(res, sampled)


class PathSamplingCell(BaseSuperNetModule):
    """The implementation of super-net cell follows `DARTS <https://github.com/quark0/darts>`__.

    When ``factory_used`` is true, it reconstructs the cell for every possible combination of operation and input index,
    because for different input index, the cell factory could instantiate different operations (e.g., with different stride).
    On export, we first have best (operation, input) pairs, the select the best ``num_ops_per_node``.

    ``loose_end`` is not supported yet, because it will cause more problems (e.g., shape mismatch).
    We assumes ``loose_end`` to be ``all`` regardless of its configuration.

    A supernet cell can't slim its own weight to fit into a sub network, which is also a known issue.
    """

    def __init__(
        self,
        op_factory: list[CellOpFactory] | dict[str, CellOpFactory],
        num_nodes: int,
        num_ops_per_node: int,
        num_predecessors: int,
        preprocessor: Any,
        postprocessor: Any,
        concat_dim: int,
        memo: dict,  # although not used here, useful in subclass
        mutate_kwargs: dict,  # same as memo
        label: str,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_ops_per_node = num_ops_per_node
        self.num_predecessors = num_predecessors
        self.preprocessor = preprocessor
        self.ops = nn.ModuleList()
        self.postprocessor = postprocessor
        self.concat_dim = concat_dim
        self.op_names: list[str] = cast(List[str], None)
        self.output_node_indices = list(range(self.num_predecessors, self.num_nodes + self.num_predecessors))

        # Create a fully-connected graph.
        # Each edge is a ModuleDict with op candidates.
        # Can not reuse LayerChoice here, because the spec, resample, export all need to be customized.
        # InputChoice is implicit in this graph.
        for i in self.output_node_indices:
            self.ops.append(nn.ModuleList())
            for k in range(i):
                # Second argument in (i, **0**, k) is always 0.
                # One-shot strategy can't handle the cases where op spec is dependent on `op_index`.
                ops, _ = create_cell_op_candidates(op_factory, i, 0, k)
                self.op_names = list(ops.keys())
                cast(nn.ModuleList, self.ops[-1]).append(nn.ModuleDict(ops))

        with label_scope(label) as self.label_scope:
            for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
                for k in range(self.num_ops_per_node):
                    op_label = f'op_{i}_{k}'
                    input_label = f'input_{i}_{k}'
                    self.add_mutable(Categorical(self.op_names, label=op_label))
                    # Need multiple here to align with the original cell.
                    self.add_mutable(CategoricalMultiple(range(i), n_chosen=1, label=input_label))

        self._sampled: dict[str, str | int] = {}

    @property
    def label(self) -> str:
        return self.label_scope.name

    def freeze(self, sample):
        raise NotImplementedError('PathSamplingCell does not support freeze.')

    def resample(self, memo):
        """Random choose one path if label is not found in memo."""
        self._sampled = {}
        new_sampled = {}
        for label, param_spec in self.simplify().items():
            if label in memo:
                if isinstance(memo[label], list) and len(memo[label]) > 1:
                    raise ValueError(f'Multi-path sampling is currently unsupported on cell: {memo[label]}')
                self._sampled[label] = memo[label]
            else:
                if isinstance(param_spec, Categorical):
                    self._sampled[label] = new_sampled[label] = random.choice(param_spec.values)
                elif isinstance(param_spec, CategoricalMultiple):
                    assert param_spec.n_chosen == 1
                    self._sampled[label] = new_sampled[label] = [random.choice(param_spec.values)]
        return new_sampled

    def export(self, memo):
        """Randomly choose one to export."""
        return self.resample(memo)

    def forward(self, *inputs: list[torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        processed_inputs: List[torch.Tensor] = preprocess_cell_inputs(self.num_predecessors, *inputs)
        states: List[torch.Tensor] = self.preprocessor(processed_inputs)
        for i, ops in enumerate(cast(Sequence[Sequence[Dict[str, nn.Module]]], self.ops), start=self.num_predecessors):
            current_state = []

            for k in range(self.num_ops_per_node):
                # Select op list based on the input chosen
                input_index = self._sampled[f'{self.label}/input_{i}_{k}'][0]  # [0] because it's a list and n_chosen=1
                op_candidates = ops[cast(int, input_index)]
                # Select op from op list based on the op chosen
                op_index = self._sampled[f'{self.label}/op_{i}_{k}']
                op = op_candidates[cast(str, op_index)]
                current_state.append(op(states[cast(int, input_index)]))

            states.append(sum(current_state))  # type: ignore

        # Always merge all
        this_cell = torch.cat(states[self.num_predecessors:], self.concat_dim)
        return self.postprocessor(this_cell, processed_inputs)

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        """
        Mutate only handles cells of specific configurations (e.g., with loose end).
        Fallback to the default mutate if the cell is not handled here.
        """
        if type(module) is Cell:
            op_factory = None  # not all the cells need to be replaced
            if module.op_candidates_factory is not None:
                op_factory = module.op_candidates_factory
                assert isinstance(op_factory, list) or isinstance(op_factory, dict), \
                    'Only support op_factory of type list or dict.'
            elif module.merge_op == 'loose_end':
                op_candidates_lc = module.ops[-1][-1]  # type: ignore
                assert isinstance(op_candidates_lc, LayerChoice)
                candidates = op_candidates_lc.candidates

                def _copy(_, __, ___, op):
                    return copy.deepcopy(op)

                if isinstance(candidates, list):
                    op_factory = [functools.partial(_copy, op=op) for op in candidates]
                elif isinstance(candidates, dict):
                    op_factory = {name: functools.partial(_copy, op=op) for name, op in candidates.items()}
                else:
                    raise ValueError(f'Unsupported type of candidates: {type(candidates)}')
            if op_factory is not None:
                return cls(
                    op_factory,
                    module.num_nodes,
                    module.num_ops_per_node,
                    module.num_predecessors,
                    module.preprocessor,
                    module.postprocessor,
                    module.concat_dim,
                    memo,
                    mutate_kwargs,
                    module.label
                )
