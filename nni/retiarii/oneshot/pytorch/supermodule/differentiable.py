# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import functools
import warnings

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.hpo_utils import ParameterSpec
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice

from .base import BaseSuperNetModule
from .operation import MixedOperation, MixedOperationSamplingPolicy
from ._valuechoice_utils import traverse_all_options


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

    def __init__(self, paths: list[tuple[str, nn.Module]], alpha: torch.Tensor, softmax: nn.Module, label: str):
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

    def forward(self, *args, **kwargs):
        """The forward of mixed layer accepts same arguments as its sub-layer."""
        op_results = torch.stack([getattr(self, op)(*args, **kwargs) for op in self.op_names])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * self._softmax(self._arch_alpha).view(*alpha_shape), 0)

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

    def __init__(self, n_candidates: int, n_chosen: int | None, alpha: torch.Tensor, softmax: nn.Module, label: str):
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

    def forward(self, inputs):
        """Forward takes a list of input candidates."""
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * self._softmax(self._arch_alpha).view(*alpha_shape), 0)

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
        """Export is also random for each leaf value choice."""
        result = {}
        for name, spec in operation.search_space_spec().items():
            if name in result:
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
