# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import functools
import logging
import warnings

from typing import Any, Dict, Sequence, List, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.hpo_utils import ParameterSpec
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice, ChoiceOf, Repeat
from nni.retiarii.nn.pytorch.api import ValueChoiceX
from nni.retiarii.nn.pytorch.cell import preprocess_cell_inputs

from .base import BaseSuperNetModule
from .operation import MixedOperation, MixedOperationSamplingPolicy
from .sampling import PathSamplingCell
from ._valuechoice_utils import traverse_all_options, dedup_inner_choices, weighted_sum

_logger = logging.getLogger(__name__)

__all__ = [
    'DifferentiableMixedLayer', 'DifferentiableMixedInput',
    'DifferentiableMixedRepeat', 'DifferentiableMixedCell',
    'MixedOpDifferentiablePolicy',
]


class GumbelSoftmax(nn.Softmax):
    """Wrapper of ``F.gumbel_softmax``. dim = -1 by default."""

    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__(dim)
        self.tau = 1
        self.hard = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gumbel_softmax(inputs, tau=self.tau, hard=self.hard, dim=self.dim)


class DifferentiableMixedLayer(BaseSuperNetModule):
    """
    Mixed layer, in which fprop is decided by a weighted sum of several layers.
    Proposed in `DARTS: Differentiable Architecture Search <https://arxiv.org/abs/1806.09055>`__.

    The weight ``alpha`` is usually learnable, and optimized on validation dataset.

    Differentiable sampling layer requires all operators returning the same shape for one input,
    as all outputs will be weighted summed to get the final output.

    Parameters
    ----------
    paths : list[tuple[str, nn.Module]]
        Layers to choose from. Each is a tuple of name, and its module.
    alpha : Tensor
        Tensor that stores the "learnable" weights.
    softmax : nn.Module
        Customizable softmax function. Usually ``nn.Softmax(-1)``.
    label : str
        Name of the choice.

    Attributes
    ----------
    op_names : str
        Operator names.
    label : str
        Name of the choice.
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self,
                 paths: list[tuple[str, nn.Module]],
                 alpha: torch.Tensor,
                 softmax: nn.Module,
                 label: str):
        super().__init__()
        self.op_names = []
        if len(alpha) != len(paths):
            raise ValueError(f'The size of alpha ({len(alpha)}) must match number of candidates ({len(paths)}).')
        for name, module in paths:
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self.label = label
        self._arch_alpha = alpha
        self._softmax = softmax

    def resample(self, memo):
        """Do nothing. Differentiable layer doesn't need resample."""
        return {}

    def export(self, memo):
        """Choose the operator with the maximum logit."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self.op_names[int(torch.argmax(self._arch_alpha).item())]}

    def search_space_spec(self):
        return {self.label: ParameterSpec(self.label, 'choice', self.op_names, (self.label, ),
                                          True, size=len(self.op_names))}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, LayerChoice):
            size = len(module)
            if module.label in memo:
                alpha = memo[module.label]
                if len(alpha) != size:
                    raise ValueError(f'Architecture parameter size of same label {module.label} conflict: {len(alpha)} vs. {size}')
            else:
                alpha = nn.Parameter(torch.randn(size) * 1E-3)  # this can be reinitialized later

            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(list(module.named_children()), alpha, softmax, module.label)

    def reduction(self, items: list[Any], weights: list[float]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, *args, **kwargs):
        """The forward of mixed layer accepts same arguments as its sub-layer."""
        all_op_results = [getattr(self, op)(*args, **kwargs) for op in self.op_names]
        return self.reduction(all_op_results, self._softmax(self._arch_alpha))

    def parameters(self, *args, **kwargs):
        """Parameters excluding architecture parameters."""
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        """Named parameters excluding architecture parameters."""
        arch = kwargs.pop('arch', False)
        for name, p in super().named_parameters(*args, **kwargs):
            if any(name == par_name for par_name in self._arch_parameter_names):
                if arch:
                    yield name, p
            else:
                if not arch:
                    yield name, p


class DifferentiableMixedInput(BaseSuperNetModule):
    """
    Mixed input. Forward returns a weighted sum of candidates.
    Implementation is very similar to :class:`DifferentiableMixedLayer`.

    Parameters
    ----------
    n_candidates : int
        Expect number of input candidates.
    n_chosen : int
        Expect numebr of inputs finally chosen.
    alpha : Tensor
        Tensor that stores the "learnable" weights.
    softmax : nn.Module
        Customizable softmax function. Usually ``nn.Softmax(-1)``.
    label : str
        Name of the choice.

    Attributes
    ----------
    label : str
        Name of the choice.
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self,
                 n_candidates: int,
                 n_chosen: int | None,
                 alpha: torch.Tensor,
                 softmax: nn.Module,
                 label: str):
        super().__init__()
        self.n_candidates = n_candidates
        if len(alpha) != n_candidates:
            raise ValueError(f'The size of alpha ({len(alpha)}) must match number of candidates ({n_candidates}).')
        if n_chosen is None:
            warnings.warn('Differentiable architecture search does not support choosing multiple inputs. Assuming one.',
                          RuntimeWarning)
            self.n_chosen = 1
        self.n_chosen = n_chosen
        self.label = label
        self._softmax = softmax

        self._arch_alpha = alpha

    def resample(self, memo):
        """Do nothing. Differentiable layer doesn't need resample."""
        return {}

    def export(self, memo):
        """Choose the operator with the top ``n_chosen`` logits."""
        if self.label in memo:
            return {}  # nothing new to export
        chosen = sorted(torch.argsort(-self._arch_alpha).cpu().numpy().tolist()[:self.n_chosen])
        if len(chosen) == 1:
            chosen = chosen[0]
        return {self.label: chosen}

    def search_space_spec(self):
        return {
            self.label: ParameterSpec(self.label, 'choice', list(range(self.n_candidates)),
                                      (self.label, ), True, size=self.n_candidates, chosen_size=self.n_chosen)
        }

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, InputChoice):
            if module.reduction not in ['sum', 'mean']:
                raise ValueError('Only input choice of sum/mean reduction is supported.')
            size = module.n_candidates
            if module.label in memo:
                alpha = memo[module.label]
                if len(alpha) != size:
                    raise ValueError(f'Architecture parameter size of same label {module.label} conflict: {len(alpha)} vs. {size}')
            else:
                alpha = nn.Parameter(torch.randn(size) * 1E-3)  # this can be reinitialized later

            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(module.n_candidates, module.n_chosen, alpha, softmax, module.label)

    def reduction(self, items: list[Any], weights: list[float]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, inputs):
        """Forward takes a list of input candidates."""
        return self.reduction(inputs, self._softmax(self._arch_alpha))

    def parameters(self, *args, **kwargs):
        """Parameters excluding architecture parameters."""
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        """Named parameters excluding architecture parameters."""
        arch = kwargs.pop('arch', False)
        for name, p in super().named_parameters(*args, **kwargs):
            if any(name == par_name for par_name in self._arch_parameter_names):
                if arch:
                    yield name, p
            else:
                if not arch:
                    yield name, p


class MixedOpDifferentiablePolicy(MixedOperationSamplingPolicy):
    """Implementes the differentiable sampling in mixed operation.

    One mixed operation can have multiple value choices in its arguments.
    Thus the ``_arch_alpha`` here is a parameter dict, and ``named_parameters``
    filters out multiple parameters with ``_arch_alpha`` as its prefix.

    When this class is asked for ``forward_argument``, it returns a distribution,
    i.e., a dict from int to float based on its weights.

    All the parameters (``_arch_alpha``, ``parameters()``, ``_softmax``) are
    saved as attributes of ``operation``, rather than ``self``,
    because this class itself is not a ``nn.Module``, and saved parameters here
    won't be optimized.
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self, operation: MixedOperation, memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> None:
        # Sampling arguments. This should have the same keys with `operation.mutable_arguments`
        operation._arch_alpha = nn.ParameterDict()
        for name, spec in operation.search_space_spec().items():
            if name in memo:
                alpha = memo[name]
                if len(alpha) != spec.size:
                    raise ValueError(f'Architecture parameter size of same label {name} conflict: {len(alpha)} vs. {spec.size}')
            else:
                alpha = nn.Parameter(torch.randn(spec.size) * 1E-3)
            operation._arch_alpha[name] = alpha

        operation.parameters = functools.partial(self.parameters, module=operation)                # bind self
        operation.named_parameters = functools.partial(self.named_parameters, module=operation)

        operation._softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))

    @staticmethod
    def parameters(module, *args, **kwargs):
        for _, p in module.named_parameters(*args, **kwargs):
            yield p

    @staticmethod
    def named_parameters(module, *args, **kwargs):
        arch = kwargs.pop('arch', False)
        for name, p in super(module.__class__, module).named_parameters(*args, **kwargs):  # pylint: disable=bad-super-call
            if any(name.startswith(par_name) for par_name in MixedOpDifferentiablePolicy._arch_parameter_names):
                if arch:
                    yield name, p
            else:
                if not arch:
                    yield name, p

    def resample(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Differentiable. Do nothing in resample."""
        return {}

    def export(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Export is argmax for each leaf value choice."""
        result = {}
        for name, spec in operation.search_space_spec().items():
            if name in memo:
                continue
            chosen_index = int(torch.argmax(cast(dict, operation._arch_alpha)[name]).item())
            result[name] = spec.values[chosen_index]
        return result

    def forward_argument(self, operation: MixedOperation, name: str) -> dict[Any, float] | Any:
        if name in operation.mutable_arguments:
            weights: dict[str, torch.Tensor] = {
                label: cast(nn.Module, operation._softmax)(alpha) for label, alpha in cast(dict, operation._arch_alpha).items()
            }
            return dict(traverse_all_options(operation.mutable_arguments[name], weights=weights))
        return operation.init_arguments[name]


class DifferentiableMixedRepeat(BaseSuperNetModule):
    """
    Implementaion of Repeat in a differentiable supernet.
    Result is a weighted sum of possible prefixes, sliced by possible depths.

    If the output is not a single tensor, it will be summed at every independant dimension.
    See :func:`weighted_sum` for details.
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self,
                 blocks: list[nn.Module],
                 depth: ChoiceOf[int],
                 softmax: nn.Module,
                 memo: dict[str, Any]):
        super().__init__()
        self.blocks = blocks
        self.depth = depth
        self._softmax = softmax
        self._space_spec: dict[str, ParameterSpec] = dedup_inner_choices([depth])
        self._arch_alpha = nn.ParameterDict()

        for name, spec in self._space_spec.items():
            if name in memo:
                alpha = memo[name]
                if len(alpha) != spec.size:
                    raise ValueError(f'Architecture parameter size of same label {name} conflict: {len(alpha)} vs. {spec.size}')
            else:
                alpha = nn.Parameter(torch.randn(spec.size) * 1E-3)
            self._arch_alpha[name] = alpha

    def resample(self, memo):
        """Do nothing."""
        return {}

    def export(self, memo):
        """Choose argmax for each leaf value choice."""
        result = {}
        for name, spec in self._space_spec.items():
            if name in memo:
                continue
            chosen_index = int(torch.argmax(self._arch_alpha[name]).item())
            result[name] = spec.values[chosen_index]
        return result

    def search_space_spec(self):
        return self._space_spec

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, Repeat) and isinstance(module.depth_choice, ValueChoiceX):
            # Only interesting when depth is mutable
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(cast(List[nn.Module], module.blocks), module.depth_choice, softmax, memo)

    def parameters(self, *args, **kwargs):
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        arch = kwargs.pop('arch', False)
        for name, p in super().named_parameters(*args, **kwargs):
            if any(name.startswith(par_name) for par_name in MixedOpDifferentiablePolicy._arch_parameter_names):
                if arch:
                    yield name, p
            else:
                if not arch:
                    yield name, p

    def reduction(self, items: list[Any], weights: list[float], depths: list[int]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, x):
        weights: dict[str, torch.Tensor] = {
            label: self._softmax(alpha) for label, alpha in self._arch_alpha.items()
        }
        depth_weights = dict(cast(List[Tuple[int, float]], traverse_all_options(self.depth, weights=weights)))

        res: list[torch.Tensor] = []
        weight_list: list[float] = []
        depths: list[int] = []
        for i, block in enumerate(self.blocks, start=1):  # start=1 because depths are 1, 2, 3, 4...
            x = block(x)
            if i in depth_weights:
                weight_list.append(depth_weights[i])
                res.append(x)
                depths.append(i)

        return self.reduction(res, weight_list, depths)


class DifferentiableMixedCell(PathSamplingCell):
    """Implementation of Cell under differentiable context.

    An architecture parameter is created on each edge of the full-connected graph.
    """

    # TODO: It inherits :class:`PathSamplingCell` to reduce some duplicated code.
    # Possibly need another refactor here.

    def __init__(
        self, op_factory, num_nodes, num_ops_per_node,
        num_predecessors, preprocessor, postprocessor, concat_dim,
        memo, mutate_kwargs, label
    ):
        super().__init__(
            op_factory, num_nodes, num_ops_per_node,
            num_predecessors, preprocessor, postprocessor,
            concat_dim, memo, mutate_kwargs, label
        )
        self._arch_alpha = nn.ParameterDict()
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            for j in range(i):
                edge_label = f'{label}/{i}_{j}'
                op = cast(List[Dict[str, nn.Module]], self.ops[i - self.num_predecessors])[j]
                if edge_label in memo:
                    alpha = memo[edge_label]
                    if len(alpha) != len(op):
                        raise ValueError(
                            f'Architecture parameter size of same label {edge_label} conflict: '
                            f'{len(alpha)} vs. {len(op)}'
                        )
                else:
                    alpha = nn.Parameter(torch.randn(len(op)) * 1E-3)
                self._arch_alpha[edge_label] = alpha

        self._softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))

    def resample(self, memo):
        """Differentiable doesn't need to resample."""
        return {}

    def export(self, memo):
        """Tricky export.

        Reference: https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model_search.py#L135
        We don't avoid selecting operations like ``none`` here, because it looks like a different search space.
        """
        exported = {}
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            # Tuple of (weight, input_index, op_name)
            all_weights: list[tuple[float, int, str]] = []
            for j in range(i):
                for k, name in enumerate(self.op_names):
                    all_weights.append((
                        float(self._arch_alpha[f'{self.label}/{i}_{j}'][k].item()),
                        j, name,
                    ))

            all_weights.sort(reverse=True)
            # We first prefer inputs from different input_index.
            # If we have got no other choices, we start to accept duplicates.
            # Therefore we gather first occurrences of distinct input_index to the front.
            first_occurrence_index: list[int] = [
                all_weights.index(                                      # The index of
                    next(filter(lambda t: t[1] == j, all_weights))      # First occurence of j
                )
                for j in range(i)                                       # For j < i
            ]
            first_occurrence_index.sort()                               # Keep them ordered too.

            all_weights = [all_weights[k] for k in first_occurrence_index] + \
                [w for j, w in enumerate(all_weights) if j not in first_occurrence_index]

            _logger.info('Sorted weights in differentiable cell export (node %d): %s', i, all_weights)

            for k in range(self.num_ops_per_node):
                # all_weights could be too short in case ``num_ops_per_node`` is too large.
                _, j, op_name = all_weights[k % len(all_weights)]
                exported[f'{self.label}/op_{i}_{k}'] = op_name
                exported[f'{self.label}/input_{i}_{k}'] = j

        return exported

    def forward(self, *inputs: list[torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        processed_inputs: list[torch.Tensor] = preprocess_cell_inputs(self.num_predecessors, *inputs)
        states: list[torch.Tensor] = self.preprocessor(processed_inputs)
        for i, ops in enumerate(cast(Sequence[Sequence[Dict[str, nn.Module]]], self.ops), start=self.num_predecessors):
            current_state = []

            for j in range(i):  # for every previous tensors
                op_results = torch.stack([op(states[j]) for op in ops[j].values()])
                alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
                edge_sum = torch.sum(op_results * self._softmax(self._arch_alpha[f'{self.label}/{i}_{j}']).view(*alpha_shape), 0)
                current_state.append(edge_sum)

            states.append(sum(current_state))  # type: ignore

        # Always merge all
        this_cell = torch.cat(states[self.num_predecessors:], self.concat_dim)
        return self.postprocessor(this_cell, processed_inputs)

    def parameters(self, *args, **kwargs):
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        arch = kwargs.pop('arch', False)
        for name, p in super().named_parameters(*args, **kwargs):
            if any(name.startswith(par_name) for par_name in MixedOpDifferentiablePolicy._arch_parameter_names):
                if arch:
                    yield name, p
            else:
                if not arch:
                    yield name, p
