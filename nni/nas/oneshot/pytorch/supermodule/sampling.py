# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
import random
from typing import Any, List, Dict, Sequence, cast

import torch
import torch.nn as nn

from nni.common.hpo_utils import ParameterSpec
from nni.nas.nn.pytorch import LayerChoice, InputChoice, Repeat, ChoiceOf, Cell
from nni.nas.nn.pytorch.choice import ValueChoiceX
from nni.nas.nn.pytorch.cell import CellOpFactory, create_cell_op_candidates, preprocess_cell_inputs

from .base import BaseSuperNetModule, sub_state_dict
from ._valuechoice_utils import evaluate_value_choice_with_dict, dedup_inner_choices, weighted_sum
from .operation import MixedOperationSamplingPolicy, MixedOperation

__all__ = [
    'PathSamplingLayer', 'PathSamplingInput',
    'PathSamplingRepeat', 'PathSamplingCell',
    'MixedOpPathSamplingPolicy'
]


class PathSamplingLayer(BaseSuperNetModule):
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

    def __init__(self, paths: list[tuple[str, nn.Module]], label: str):
        super().__init__()
        self.op_names = []
        for name, module in paths:
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self._sampled: list[str] | str | None = None  # sampled can be either a list of indices or an index
        self.label = label

    def resample(self, memo):
        """Random choose one path if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            self._sampled = random.choice(self.op_names)
        return {self.label: self._sampled}

    def export(self, memo):
        """Random choose one name if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: random.choice(self.op_names)}

    def search_space_spec(self):
        return {self.label: ParameterSpec(self.label, 'choice', self.op_names, (self.label, ),
                                          True, size=len(self.op_names))}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, LayerChoice):
            return cls(list(module.named_children()), module.label)

    def reduction(self, items: list[Any], sampled: list[Any]):
        """Override this to implement customized reduction."""
        return weighted_sum(items)

    def _save_module_to_state_dict(self, destination, prefix, keep_vars):
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        for samp in sampled:
            module = getattr(self, str(samp))
            if module is not None:
                sub_state_dict(module, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, *args, **kwargs):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        # str(samp) is needed here because samp can sometimes be integers, but attr are always str
        res = [getattr(self, str(samp))(*args, **kwargs) for samp in sampled]
        return self.reduction(res, sampled)


class PathSamplingInput(BaseSuperNetModule):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled input indices.
    """

    def __init__(self, n_candidates: int, n_chosen: int, reduction_type: str, label: str):
        super().__init__()
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction_type = reduction_type
        self._sampled: list[int] | int | None = None
        self.label = label

    def _random_choose_n(self):
        sampling = list(range(self.n_candidates))
        random.shuffle(sampling)
        sampling = sorted(sampling[:self.n_chosen])
        if len(sampling) == 1:
            return sampling[0]
        else:
            return sampling

    def resample(self, memo):
        """Random choose one path / multiple paths if label is not found in memo.
        If one path is selected, only one integer will be in ``self._sampled``.
        If multiple paths are selected, a list will be in ``self._sampled``.
        """
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            self._sampled = self._random_choose_n()
        return {self.label: self._sampled}

    def export(self, memo):
        """Random choose one name if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self._random_choose_n()}

    def search_space_spec(self):
        return {
            self.label: ParameterSpec(self.label, 'choice', list(range(self.n_candidates)),
                                      (self.label, ), True, size=self.n_candidates, chosen_size=self.n_chosen)
        }

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, InputChoice):
            if module.reduction not in ['sum', 'mean', 'concat']:
                raise ValueError('Only input choice of sum/mean/concat reduction is supported.')
            if module.n_chosen is None:
                raise ValueError('n_chosen is None is not supported yet.')
            return cls(module.n_candidates, module.n_chosen, module.reduction, module.label)

    def reduction(self, items: list[Any], sampled: list[Any]) -> Any:
        """Override this to implement customized reduction."""
        if len(items) == 1:
            return items[0]
        else:
            if self.reduction_type == 'sum':
                return sum(items)
            elif self.reduction_type == 'mean':
                return sum(items) / len(items)
            elif self.reduction_type == 'concat':
                return torch.cat(items, 1)
            raise ValueError(f'Unsupported reduction type: {self.reduction_type}')

    def forward(self, input_tensors):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        if len(input_tensors) != self.n_candidates:
            raise ValueError(f'Expect {self.n_candidates} input tensors, found {len(input_tensors)}.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled
        res = [input_tensors[samp] for samp in sampled]
        return self.reduction(res, sampled)


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
        space_spec = operation.search_space_spec()
        for label in space_spec:
            if label in memo:
                result[label] = memo[label]
            else:
                result[label] = random.choice(space_spec[label].values)

        # composits to kwargs
        # example: result = {"exp_ratio": 3}, self._sampled = {"in_channels": 48, "out_channels": 96}
        self._sampled = {}
        for key, value in operation.mutable_arguments.items():
            self._sampled[key] = evaluate_value_choice_with_dict(value, result)

        return result

    def export(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Export is also random for each leaf value choice."""
        result = {}
        space_spec = operation.search_space_spec()
        for label in space_spec:
            if label not in memo:
                result[label] = random.choice(space_spec[label].values)
        return result

    def forward_argument(self, operation: MixedOperation, name: str) -> Any:
        # NOTE: we don't support sampling a list here.
        if self._sampled is None:
            raise ValueError('Need to call resample() before running forward')
        if name in operation.mutable_arguments:
            return self._sampled[name]
        return operation.init_arguments[name]


class PathSamplingRepeat(BaseSuperNetModule):
    """
    Implementaion of Repeat in a path-sampling supernet.
    Samples one / some of the prefixes of the repeated blocks.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled depth.
    """

    def __init__(self, blocks: list[nn.Module], depth: ChoiceOf[int]):
        super().__init__()
        self.blocks: Any = blocks
        self.depth = depth
        self._space_spec: dict[str, ParameterSpec] = dedup_inner_choices([depth])
        self._sampled: list[int] | int | None = None

    def resample(self, memo):
        """Since depth is based on ValueChoice, we only need to randomly sample every leaf value choices."""
        result = {}
        for label in self._space_spec:
            if label in memo:
                result[label] = memo[label]
            else:
                result[label] = random.choice(self._space_spec[label].values)

        self._sampled = evaluate_value_choice_with_dict(self.depth, result)

        return result

    def export(self, memo):
        """Random choose one if every choice not in memo."""
        result = {}
        for label in self._space_spec:
            if label not in memo:
                result[label] = random.choice(self._space_spec[label].values)
        return result

    def search_space_spec(self):
        return self._space_spec

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, Repeat) and isinstance(module.depth_choice, ValueChoiceX):
            # Only interesting when depth is mutable
            return cls(cast(List[nn.Module], module.blocks), module.depth_choice)

    def reduction(self, items: list[Any], sampled: list[Any]):
        """Override this to implement customized reduction."""
        return weighted_sum(items)

    def _save_module_to_state_dict(self, destination, prefix, keep_vars):
        sampled: Any = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        for cur_depth, (name, module) in enumerate(self.blocks.named_children(), start=1):
            if module is not None:
                sub_state_dict(module, destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
            if not any(d > cur_depth for d in sampled):
                break

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
        return self.reduction(res, sampled)


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
            for k in range(i + self.num_predecessors):
                # Second argument in (i, **0**, k) is always 0.
                # One-shot strategy can't handle the cases where op spec is dependent on `op_index`.
                ops, _ = create_cell_op_candidates(op_factory, i, 0, k)
                self.op_names = list(ops.keys())
                cast(nn.ModuleList, self.ops[-1]).append(nn.ModuleDict(ops))

        self.label = label

        self._sampled: dict[str, str | int] = {}

    def search_space_spec(self) -> dict[str, ParameterSpec]:
        # TODO: Recreating the space here.
        # The spec should be moved to definition of Cell itself.
        space_spec = {}
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            for k in range(self.num_ops_per_node):
                op_label = f'{self.label}/op_{i}_{k}'
                input_label = f'{self.label}/input_{i}_{k}'
                space_spec[op_label] = ParameterSpec(op_label, 'choice', self.op_names, (op_label,), True, size=len(self.op_names))
                space_spec[input_label] = ParameterSpec(input_label, 'choice', list(range(i)), (input_label, ), True, size=i)
        return space_spec

    def resample(self, memo):
        """Random choose one path if label is not found in memo."""
        self._sampled = {}
        new_sampled = {}
        for label, param_spec in self.search_space_spec().items():
            if label in memo:
                assert not isinstance(memo[label], list), 'Multi-path sampling is currently unsupported on cell.'
                self._sampled[label] = memo[label]
            else:
                self._sampled[label] = new_sampled[label] = random.choice(param_spec.values)
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
                input_index = self._sampled[f'{self.label}/input_{i}_{k}']
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
        if isinstance(module, Cell):
            op_factory = None  # not all the cells need to be replaced
            if module.op_candidates_factory is not None:
                op_factory = module.op_candidates_factory
                assert isinstance(op_factory, list) or isinstance(op_factory, dict), \
                    'Only support op_factory of type list or dict.'
            elif module.merge_op == 'loose_end':
                op_candidates_lc = module.ops[-1][-1]  # type: ignore
                assert isinstance(op_candidates_lc, LayerChoice)
                op_factory = {  # create a factory
                    name: lambda _, __, ___: copy.deepcopy(op_candidates_lc[name])
                    for name in op_candidates_lc.names
                }
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
