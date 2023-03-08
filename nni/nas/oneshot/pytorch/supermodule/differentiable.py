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

from nni.mutable import MutableExpression, Mutable, Categorical
from nni.nas.nn.pytorch import LayerChoice, InputChoice, Repeat
from nni.nas.nn.pytorch.cell import preprocess_cell_inputs

from .base import BaseSuperNetModule
from .operation import MixedOperation, MixedOperationSamplingPolicy
from .sampling import PathSamplingCell
from ._expression_utils import traverse_all_options, weighted_sum

_logger = logging.getLogger(__name__)

__all__ = [
    'DifferentiableMixedLayer', 'DifferentiableMixedInput',
    'DifferentiableMixedRepeat', 'DifferentiableMixedCell',
    'MixedOpDifferentiablePolicy', 'GumbelSoftmax',
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


class DifferentiableMixedLayer(LayerChoice, BaseSuperNetModule):
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
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self,
                 paths: list[nn.Module] | dict[str, nn.Module],
                 alpha: torch.Tensor,
                 softmax: nn.Module,
                 label: str):
        super().__init__(paths, label=label)
        if len(alpha) != len(paths):
            raise ValueError(f'The size of alpha ({len(alpha)}) must match number of candidates ({len(paths)}).')
        self._arch_alpha = alpha
        self._softmax = softmax

    def resample(self, memo):
        """Do nothing. Differentiable layer doesn't need resample."""
        return {}

    def export(self, memo):
        """Choose the operator with the maximum logit."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self.names[int(torch.argmax(self._arch_alpha).item())]}

    def export_probs(self, memo):
        if self.label in memo:
            return {}
        weights = self._softmax(self._arch_alpha).cpu().tolist()
        return {self.label: dict(zip(self.names, weights))}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) is LayerChoice:  # must be exactly LayerChoice
            if module.label not in memo:
                raise KeyError(f'LayerChoice {module.label} not found in memo.')
            alpha = memo[module.label]
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(module.candidates, alpha, softmax, module.label)

    def _reduction(self, items: list[Any], weights: list[float]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, *args, **kwargs):
        """The forward of mixed layer accepts same arguments as its sub-layer."""
        all_op_results = [self[op](*args, **kwargs) for op in self.names]
        return self._reduction(all_op_results, self._softmax(self._arch_alpha))

    def arch_parameters(self):
        """Iterate over architecture parameters. Not recursive."""
        for name, p in self.named_parameters():
            if any(name == par_name for par_name in self._arch_parameter_names):
                yield p


class DifferentiableMixedInput(InputChoice, BaseSuperNetModule):
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
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    def __init__(self,
                 n_candidates: int,
                 n_chosen: int | None,
                 alpha: torch.Tensor,
                 softmax: nn.Module,
                 label: str):
        if n_chosen is None:
            warnings.warn('Differentiable architecture search does not support choosing multiple inputs. Assuming one.',
                          RuntimeWarning)
            n_chosen = 1
        super().__init__(n_candidates, n_chosen=n_chosen, label=label)
        if len(alpha) != n_candidates:
            raise ValueError(f'The size of alpha ({len(alpha)}) must match number of candidates ({n_candidates}).')
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
        return {self.label: chosen}

    def export_probs(self, memo):
        if self.label in memo:
            return {}
        weights = self._softmax(self._arch_alpha).cpu().tolist()
        return {self.label: dict(enumerate(weights))}

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) == InputChoice:  # must be exactly InputChoice
            module = cast(InputChoice, module)
            if module.reduction not in ['sum', 'mean']:
                raise ValueError('Only input choice of sum/mean reduction is supported.')
            if module.label not in memo:
                raise KeyError(f'InputChoice {module.label} not found in memo.')
            alpha = memo[module.label]
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(module.n_candidates, module.n_chosen, alpha, softmax, module.label)

    def _reduction(self, items: list[Any], weights: list[float]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, inputs):
        """Forward takes a list of input candidates."""
        return self._reduction(inputs, self._softmax(self._arch_alpha))

    def arch_parameters(self):
        """Iterate over architecture parameters. Not recursive."""
        for name, p in self.named_parameters():
            if any(name == par_name for par_name in self._arch_parameter_names):
                yield p


class MixedOpDifferentiablePolicy(MixedOperationSamplingPolicy):
    """Implements the differentiable sampling in mixed operation.

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
        for name in operation.simplify():
            if name not in memo:
                raise KeyError(f'Argument {name} not found in memo.')
            operation._arch_alpha[str(name)] = memo[name]

        operation.arch_parameters = functools.partial(self.arch_parameters, module=operation)

        operation._softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))

    @staticmethod
    def arch_parameters(module):
        """Iterate over architecture parameters. Not recursive."""
        for name, p in module.named_parameters():
            if any(name.startswith(par_name) for par_name in MixedOpDifferentiablePolicy._arch_parameter_names):
                yield p

    def resample(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Differentiable. Do nothing in resample."""
        return {}

    def export(self, operation: MixedOperation, memo: dict[str, Any]) -> dict[str, Any]:
        """Export is argmax for each leaf value choice."""
        result = {}
        for name, spec in operation.simplify().items():
            if name in memo:
                continue
            chosen_index = int(torch.argmax(cast(dict, operation._arch_alpha)[name]).item())
            result[name] = cast(Categorical, spec).values[chosen_index]
        return result

    def export_probs(self, operation: MixedOperation, memo: dict[str, Any]):
        """Export the weight for every leaf value choice."""
        ret = {}
        for name, spec in operation.simplify().items():
            if name in memo:
                continue
            weights = operation._softmax(operation._arch_alpha[name]).cpu().tolist()  # type: ignore
            ret.update({name: dict(zip(cast(Categorical, spec).values, weights))})
        return ret

    def forward_argument(self, operation: MixedOperation, name: str) -> dict[Any, float] | Any:
        if name in operation.mutable_arguments:
            weights: dict[str, torch.Tensor] = {
                label: cast(nn.Module, operation._softmax)(alpha) for label, alpha in cast(dict, operation._arch_alpha).items()
            }
            return dict(traverse_all_options(operation.mutable_arguments[name], weights=weights))
        return operation.init_arguments[name]


class DifferentiableMixedRepeat(Repeat, BaseSuperNetModule):
    """
    Implementation of Repeat in a differentiable supernet.
    Result is a weighted sum of possible prefixes, sliced by possible depths.

    If the output is not a single tensor, it will be summed at every independant dimension.
    See :func:`weighted_sum` for details.
    """

    _arch_parameter_names: list[str] = ['_arch_alpha']

    depth_choice: MutableExpression[int]

    def __init__(self,
                 blocks: list[nn.Module],
                 depth: MutableExpression[int],
                 softmax: nn.Module,
                 alphas: dict[str, Any]):
        assert isinstance(depth, Mutable)
        super().__init__(blocks, depth)
        self._softmax = softmax
        self._arch_alpha = nn.ParameterDict(alphas)

    def resample(self, memo):
        """Do nothing."""
        return {}

    def export(self, memo):
        """Choose argmax for each leaf value choice."""
        result = {}
        for name, spec in self.depth_choice.simplify().items():
            if name in memo:
                continue
            chosen_index = int(torch.argmax(self._arch_alpha[name]).item())
            result[name] = cast(Categorical, spec).values[chosen_index]
        return result

    def export_probs(self, memo):
        """Export the weight for every leaf value choice."""
        ret = {}
        for name, spec in self.depth_choice.simplify().items():
            if name in memo:
                continue
            weights = self._softmax(self._arch_alpha[name]).cpu().tolist()
            ret.update({name: dict(zip(cast(Categorical, spec).values, weights))})
        return ret

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) == Repeat and isinstance(module.depth_choice, Mutable):  # Repeat and depth is mutable
            # Only interesting when depth is mutable
            module = cast(Repeat, module)
            alphas = {}
            for name in cast(Mutable, module.depth_choice).simplify():
                if name not in memo:
                    raise KeyError(f'Mutable depth "{name}" not found in memo')
                alphas[name] = memo[name]
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            return cls(list(module.blocks), cast(MutableExpression[int], module.depth_choice), softmax, alphas)

    def arch_parameters(self):
        """Iterate over architecture parameters. Not recursive."""
        for name, p in self.named_parameters():
            if any(name.startswith(par_name) for par_name in self._arch_parameter_names):
                yield p

    def _reduction(self, items: list[Any], weights: list[float], depths: list[int]) -> Any:
        """Override this for customized reduction."""
        # Use weighted_sum to handle complex cases where sequential output is not a single tensor
        return weighted_sum(items, weights)

    def forward(self, x):
        weights: dict[str, torch.Tensor] = {
            label: self._softmax(alpha) for label, alpha in self._arch_alpha.items()
        }
        depth_weights = dict(cast(List[Tuple[int, float]], traverse_all_options(self.depth_choice, weights=weights)))

        res: list[torch.Tensor] = []
        weight_list: list[float] = []
        depths: list[int] = []
        for i, block in enumerate(self.blocks, start=1):  # start=1 because depths are 1, 2, 3, 4...
            x = block(x)
            if i in depth_weights:
                weight_list.append(depth_weights[i])
                res.append(x)
                depths.append(i)

        return self._reduction(res, weight_list, depths)


class DifferentiableMixedCell(PathSamplingCell):
    """Implementation of Cell under differentiable context.

    Similar to PathSamplingCell, this cell only handles cells of specific kinds (e.g., with loose end).

    An architecture parameter is created on each edge of the full-connected graph.
    """

    # TODO: It inherits :class:`PathSamplingCell` to reduce some duplicated code.
    # Possibly need another refactor here.

    _arch_parameter_names: list[str] = ['_arch_alpha']

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
                edge_label = f'{self.label}/{i}_{j}'
                # Some parameters still need to be created here inside.
                # We should avoid conflict with "outside parameters".
                memo_label = edge_label + '/in_cell'
                op = cast(List[Dict[str, nn.Module]], self.ops[i - self.num_predecessors])[j]
                if memo_label in memo:
                    alpha = memo[memo_label]
                    if len(alpha) != len(op) + 1:
                        if len(alpha) != len(op):
                            raise ValueError(
                                f'Architecture parameter size of same label {edge_label} conflict: '
                                f'{len(alpha)} vs. {len(op)}'
                            )
                        warnings.warn(
                            f'Architecture parameter size {len(alpha)} is not same as expected: {len(op) + 1}. '
                            'This is likely due to the label being shared by a LayerChoice inside the cell and outside.',
                            UserWarning
                        )
                else:
                    # +1 to emulate the input choice.
                    alpha = nn.Parameter(torch.randn(len(op) + 1) * 1E-3)
                    memo[memo_label] = alpha
                self._arch_alpha[edge_label] = alpha

        self._softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))

    def resample(self, memo):
        """Differentiable doesn't need to resample."""
        return {}

    def export_probs(self, memo):
        """When export probability, we follow the structure in arch alpha."""
        ret = {}
        for name, parameter in self._arch_alpha.items():
            if name in memo:
                continue
            weights = self._softmax(parameter).cpu().tolist()
            ret.update({name: dict(zip(self.op_names, weights))})
        return ret

    def export(self, memo):
        """Tricky export.

        Reference: https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model_search.py#L135
        """
        exported = {}
        for i in range(self.num_predecessors, self.num_nodes + self.num_predecessors):
            # If label already exists, no need to re-export.
            if all(f'{self.label}/op_{i}_{k}' in memo and f'{self.label}/input_{i}_{k}' in memo for k in range(self.num_ops_per_node)):
                continue

            # Tuple of (weight, input_index, op_name)
            all_weights: list[tuple[float, int, str]] = []
            for j in range(i):
                for k, name in enumerate(self.op_names):
                    # The last appended weight is automatically skipped in export.
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

            _logger.info('Sorted weights in differentiable cell export (%s cell, node %d): %s', self.label, i, all_weights)

            for k in range(self.num_ops_per_node):
                # all_weights could be too short in case ``num_ops_per_node`` is too large.
                _, j, op_name = all_weights[k % len(all_weights)]
                exported[f'{self.label}/op_{i}_{k}'] = op_name
                exported[f'{self.label}/input_{i}_{k}'] = [j]

        return exported

    def forward(self, *inputs: list[torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, ...] | torch.Tensor:
        processed_inputs: list[torch.Tensor] = preprocess_cell_inputs(self.num_predecessors, *inputs)
        states: list[torch.Tensor] = self.preprocessor(processed_inputs)
        for i, ops in enumerate(cast(Sequence[Sequence[Dict[str, nn.Module]]], self.ops), start=self.num_predecessors):
            current_state = []

            for j in range(i):  # for every previous tensors
                op_results = torch.stack([op(states[j]) for op in ops[j].values()])
                alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)  # (-1, 1, 1, 1, 1, ...)
                op_weights = self._softmax(self._arch_alpha[f'{self.label}/{i}_{j}'])
                if op_weights.size(0) == op_results.size(0) + 1:
                    # concatenate with a zero operation, indicating this path is not chosen at all.
                    op_results = torch.cat((op_results, torch.zeros_like(op_results[:1])), 0)
                edge_sum = torch.sum(op_results * op_weights.view(*alpha_shape), 0)
                current_state.append(edge_sum)

            states.append(sum(current_state))  # type: ignore

        # Always merge all
        this_cell = torch.cat(states[self.num_predecessors:], self.concat_dim)
        return self.postprocessor(this_cell, processed_inputs)

    def arch_parameters(self):
        """Iterate over architecture parameters. Not recursive."""
        for name, p in self.named_parameters():
            if any(name.startswith(par_name) for par_name in self._arch_parameter_names):
                yield p
