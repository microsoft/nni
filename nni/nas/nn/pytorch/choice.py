# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Can't have annotations on because PyTorch JIT doesn't support it.
# from __future__ import annotations

import functools
import warnings
from typing import (Any, Iterator, List, Optional, Dict, Union, Tuple, cast)
from typing_extensions import Literal

import torch
import torch.nn as nn
from nni.mutable import Categorical, CategoricalMultiple, Sample, SampleValidationError, ensure_frozen

from .base import MutableModule, recursive_freeze

__all__ = [
    # APIs
    'LayerChoice',
    'InputChoice',
    'ValueChoice',
    'MutationAnchor',

    # Fixed module
    'ChosenInputs',

    # Type utils
    'ReductionType',
]


class ValueChoice(Categorical):
    """For backward compatibility only. Please use :class:`nni.mutable.Categorical` instead."""

    @functools.wraps(Categorical.__init__)
    def __init__(self, *args, **kwargs):
        warnings.warn('ValueChoice is deprecated, please use `nni.choice` instead', DeprecationWarning)
        super().__init__(*args, **kwargs)

    @property
    def candidates(self) -> list:
        return self.values


class LayerChoice(MutableModule):
    """
    Layer choice selects one of the ``candidates``, then apply it on inputs and return results.

    It allows users to put several candidate operations (e.g., PyTorch modules), one of them is chosen in each explored model.

    *New in v2.2:* Layer choice can be nested.

    Parameters
    ----------
    candidates : list of nn.Module or OrderedDict
        A module list to be selected from.
    weights : list of float
        Prior distribution used in random sampling.
    label : str
        Identifier of the layer choice.

    Attributes
    ----------
    length : int
        Deprecated. Number of ops to choose from. ``len(layer_choice)`` is recommended.
    names : list of str
        Names of candidates.
    choices : list of Module
        Deprecated. A list of all candidate modules in the layer choice module.
        ``list(layer_choice)`` is recommended, which will serve the same purpose.

    Examples
    --------

    ::

        # import nni.nas.nn.pytorch as nn
        # declared in `__init__` method
        self.layer = nn.LayerChoice([
            ops.PoolBN('max', channels, 3, stride, 1),
            ops.SepConv(channels, channels, 3, stride, 1),
            nn.Identity()
        ])
        # invoked in `forward` method
        out = self.layer(x)

    Notes
    -----
    ``candidates`` can be a list of modules or a ordered dict of named modules, for example,

    .. code-block:: python

        self.op_choice = LayerChoice(OrderedDict([
            ("conv3x3", nn.Conv2d(3, 16, 128)),
            ("conv5x5", nn.Conv2d(5, 16, 128)),
            ("conv7x7", nn.Conv2d(7, 16, 128))
        ]))

    Elements in layer choice can be modified or deleted. Use ``del self.op_choice["conv5x5"]`` or
    ``self.op_choice[1] = nn.Conv3d(...)``. Adding more choices is not supported yet.
    """

    def __init__(self, candidates: Union[Dict[str, nn.Module], List[nn.Module]], *,
                 weights: Optional[List[float]] = None, label: Optional[str] = None):
        super().__init__()

        _names, _modules = self._init_names(candidates)
        for name, module in zip(_names, _modules):
            self.add_module(str(name), module)

        self.choice = self._inner_choice(_names, weights=weights, label=label)
        self.add_mutable(self.choice)
        self._dry_run_choice = ensure_frozen(self.choice)

        # Names are kept as original types. They need to be converted to str for getattr.
        self.names: Union[List[str], List[int]] = _names

    @torch.jit.unused
    @property
    def label(self) -> str:
        return self.choice.label

    @torch.jit.unused
    @property
    def candidates(self) -> Union[Dict[str, nn.Module], List[nn.Module]]:
        """Restore the ``candidates`` parameters passed to the constructor.
        Useful when creating a new layer choices based on this one.
        """
        if all(isinstance(name, int) for name in self.names) and self.names == list(range(len(self))):
            return list(self)
        else:
            return {cast(str, name): self[name] for name in self.names}

    @staticmethod
    def _inner_choice(names: List[str], weights: Optional[List[float]], label: Optional[str]) -> Categorical:
        return Categorical(names, weights=weights, label=label)

    @staticmethod
    def _init_names(candidates: Union[Dict[str, nn.Module], List[nn.Module]]) -> Tuple[List[str], List[nn.Module]]:
        names, modules = [], []
        if isinstance(candidates, dict):
            for name, module in candidates.items():
                assert name not in ["length", "reduction", "return_mask", "_key", "key", "names"], \
                    "Please don't use a reserved name '{}' for your module.".format(name)
                if not isinstance(name, str):
                    raise TypeError(f'Key of candidates must be str, got {type(name)}.')
                names.append(name)
                modules.append(module)
        elif isinstance(candidates, list):
            for i, module in enumerate(candidates):
                names.append(i)
                modules.append(module)
        else:
            raise TypeError("Unsupported candidates type: {}".format(type(candidates)))

        return names, modules

    def check_contains(self, sample: Sample) -> Optional[SampleValidationError]:
        exception = self.choice.check_contains(sample)
        if exception is not None:
            return exception

        sample_val = self.choice.freeze(sample)
        module = self[sample_val]
        if isinstance(module, MutableModule):
            exception = module.check_contains(sample)
            if exception is not None:
                exception.paths.append(sample_val)
                return exception
        else:
            for name, submodule in MutableModule.named_mutable_descendants(module):  # type: ignore
                exception = submodule.check_contains(sample)
                if exception is not None:
                    exception.paths.append(name)
                    exception.paths.append(sample_val)
                    return exception

        return None

    def freeze(self, sample: Sample) -> nn.Module:
        self.validate(sample)
        sample_val = self.choice.freeze(sample)
        return recursive_freeze(self[sample_val], sample)[0]

    @classmethod
    def create_fixed_module(cls, sample: dict, candidates: Union[Dict[str, nn.Module], List[nn.Module]], *,
                            weights: Optional[List[float]] = None, label: Optional[str] = None):
        names, _ = cls._init_names(candidates)
        chosen = cls._inner_choice(names, weights, label).freeze(sample)
        if isinstance(candidates, list):
            result = candidates[int(chosen)]
        else:
            result = candidates[chosen]
        return result

    def __getitem__(self, idx: Union[int, str]) -> nn.Module:
        if idx not in self.names:
            raise KeyError(f'{idx!r} is not found in {self.names!r}.')
        return cast(nn.Module, self._modules[str(idx)])

    def __setitem__(self, idx, module):
        if idx not in self.names:
            raise KeyError(f'{idx!r} is not found in {self.names!r}. Note we disallow adding new choices to LayerChoice.')
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        raise RuntimeError('Deleting choices from LayerChoice is not supported yet.')

    def __len__(self):
        return len(self.names)

    def __iter__(self) -> Iterator[nn.Module]:
        return map(lambda name: cast(nn.Module, self._modules[str(name)]), self.names)

    def forward(self, x):
        # The input argument can be arbitrary positional / keyword arguments,
        # but JIT is unhappy with the unrestricted cases.

        # The forward of layer choice is simply running the first candidate module.
        # It shouldn't be called directly by users in most cases.
        for name, child_module in self.named_children():
            # Explicitly cast str here to make JIT happy
            if str(name) == str(self._dry_run_choice):
                return child_module(x)
        raise RuntimeError('dry_run_choice is not available. This should not happen.')

    def extra_repr(self):
        return f'label={self.label!r}'


ReductionType = Literal['mean', 'concat', 'sum', 'none']


class InputChoice(MutableModule):
    """
    Input choice selects ``n_chosen`` inputs from ``choose_from`` (contains ``n_candidates`` keys).

    It is mainly for choosing (or trying) different connections. It takes several tensors and chooses ``n_chosen`` tensors from them.
    When specific inputs are chosen, ``InputChoice`` will become :class:`ChosenInputs`.

    Use ``reduction`` to specify how chosen inputs are reduced into one output. A few options are:

    * ``none``: do nothing and return the list directly.
    * ``sum``: summing all the chosen inputs.
    * ``mean``: taking the average of all chosen inputs.
    * ``concat``: concatenate all chosen inputs at dimension 1.

    We don't support customizing reduction yet.

    Parameters
    ----------
    n_candidates : int
        Number of inputs to choose from. It is required.
    n_chosen : int
        Recommended inputs to choose. If None, mutator is instructed to select any.
    reduction : str
        ``mean``, ``concat``, ``sum`` or ``none``.
    weights : list of float
        Prior distribution used in random sampling.
    label : str
        Identifier of the input choice.

    Examples
    --------
    ::

        # import nni.nas.nn.pytorch as nn
        # declared in `__init__` method
        self.input_switch = nn.InputChoice(n_chosen=1)
        # invoked in `forward` method, choose one from the three
        out = self.input_switch([tensor1, tensor2, tensor3])
    """

    @classmethod
    def create_fixed_module(cls, sample: dict, n_candidates: int, n_chosen: Optional[int] = 1,
                            reduction: ReductionType = 'sum', *,
                            weights: Optional[List[float]] = None, label: Optional[str] = None, **kwargs):
        sample_val = cls._inner_choice(n_candidates, n_chosen, weights, label).freeze(sample)
        return ChosenInputs(sample_val, reduction=reduction)

    @staticmethod
    def _inner_choice(n_candidates: int, n_chosen: Optional[int],
                      weights: Optional[List[float]], label: Optional[str]) -> CategoricalMultiple:
        return CategoricalMultiple(range(n_candidates), n_chosen=n_chosen, weights=weights, label=label)

    def __init__(self, n_candidates: int, n_chosen: Optional[int] = 1,
                 reduction: ReductionType = 'sum', *,
                 weights: Optional[List[float]] = None, label: Optional[str] = None):
        super().__init__()
        if reduction not in ['mean', 'concat', 'sum', 'none']:
            raise ValueError('reduction must be one of mean, concat, sum, none')
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction: ReductionType = reduction
        self.weights = weights or [1 / n_candidates for _ in range(n_candidates)]

        self.choice = self._inner_choice(n_candidates, n_chosen, weights, label)
        self.add_mutable(self.choice)

        self._dry_run_choice: Union[int, List[int]] = ensure_frozen(self.choice)

    @torch.jit.unused
    @property
    def label(self) -> str:
        return self.choice.label

    def freeze(self, sample: Dict[str, Any]) -> nn.Module:
        self.validate(sample)
        sample_val: Union[int, List[int]] = self.choice.freeze(sample)
        return ChosenInputs(sample_val, reduction=self.reduction)

    def forward(self, candidate_inputs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        The forward of input choice is simply the first item of ``candidate_inputs``.
        It shouldn't be called directly by users in most cases.
        """
        if isinstance(self._dry_run_choice, int):
            return candidate_inputs[self._dry_run_choice]
        else:
            return self._tensor_reduction(candidate_inputs)

    def extra_repr(self):
        return f'n_candidates={self.n_candidates}, n_chosen={self.n_chosen}, reduction={repr(self.reduction)}, label={repr(self.label)})'

    @torch.jit.ignore  # type: ignore
    def _tensor_reduction(self, candidate_inputs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        return ChosenInputs._tensor_reduction(self.reduction, [candidate_inputs[idx] for idx in self._dry_run_choice])  # type: ignore


class ChosenInputs(nn.Module):
    """
    A module that chooses from a tensor list and outputs a reduced tensor.
    The already-chosen version of InputChoice.

    When forward, ``chosen`` will be used to select inputs from ``candidate_inputs``,
    and ``reduction`` will be used to choose from those inputs to form a tensor.

    Attributes
    ----------
    chosen : list of int
        Indices of chosen inputs.
    reduction : ``mean`` | ``concat`` | ``sum`` | ``none``
        How to reduce the inputs when multiple are selected.
    """

    def __init__(self, chosen: Union[List[int], int], reduction: ReductionType):
        super().__init__()
        self.chosen = chosen if isinstance(chosen, list) else [chosen]
        self.reduction = reduction

    def forward(self, candidate_inputs: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute the reduced input based on ``chosen`` and ``reduction``.
        """
        return self._tensor_reduction(self.reduction, [candidate_inputs[i] for i in self.chosen])  # type: ignore

    @staticmethod
    def _tensor_reduction(reduction_type: str, tensor_list: List[torch.Tensor]) -> Union[List[torch.Tensor], torch.Tensor, None]:
        if reduction_type == 'none':
            return tensor_list
        if not tensor_list:
            return None  # empty. return None for now
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == 'sum':
            return cast(torch.Tensor, sum(tensor_list))
        if reduction_type == 'mean':
            return cast(torch.Tensor, sum(tensor_list) / len(tensor_list))
        if reduction_type == 'concat':
            return torch.cat(tensor_list, dim=1)
        raise ValueError(f'Unrecognized reduction policy: "{reduction_type}"')


class MutationAnchor(MutableModule):
    """
    The API that creates an empty module for later mutations.
    For advanced usages only.
    """

    def __init__(self, *, label: str, **kwargs):
        super().__init__()
        self.label = label
        self.kwargs = kwargs

    def forward(self, x):
        """
        Forward of placeholder is not meaningful.
        It returns input directly.
        """
        return x
