# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Optional, List, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

from nni.common.hpo_utils import ParameterSpec
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice

from .base import BaseSuperNetModule
from ._valuechoice_utils import evaluate_value_choice_with_dict
from .operation import MixedOperationSamplingPolicy, MixedOperation


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

    def __init__(self, paths: List[Tuple[str, nn.Module]], label: str):
        super().__init__()
        self.op_names = []
        for name, module in paths:
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self._sampled: Optional[Union[List[str], str]] = None  # sampled can be either a list of indices or an index
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

    def forward(self, *args, **kwargs):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        # str(samp) is needed here because samp can sometimes be integers, but attr are always str
        res = [getattr(self, str(samp))(*args, **kwargs) for samp in sampled]
        if len(res) == 1:
            return res[0]
        else:
            return sum(res)


class PathSamplingInput(BaseSuperNetModule):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled input indices.
    """

    def __init__(self, n_candidates: int, n_chosen: int, reduction: str, label: str):
        super().__init__()
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
        self.reduction = reduction
        self._sampled: Optional[Union[List[int], int]] = None
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

    def forward(self, input_tensors):
        if self._sampled is None:
            raise RuntimeError('At least one path needs to be sampled before fprop.')
        if len(input_tensors) != self.n_candidates:
            raise ValueError(f'Expect {self.n_candidates} input tensors, found {len(input_tensors)}.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled
        res = [input_tensors[samp] for samp in sampled]
        if len(res) == 1:
            return res[0]
        else:
            if self.reduction == 'sum':
                return sum(res)
            elif self.reduction == 'mean':
                return sum(res) / len(res)
            elif self.reduction == 'concat':
                return torch.cat(res, 1)


class MixedOpPathSamplingPolicy(MixedOperationSamplingPolicy):
    """Implementes the path sampling in mixed operation.

    One mixed operation can have multiple value choices in its arguments.
    Each value choice can be further decomposed into "leaf value choices".
    We sample the leaf nodes, and composits them into the values on arguments.
    """

    def __init__(self, operation: MixedOperation, memo: Dict[str, Any], mutate_kwargs: Dict[str, Any]) -> None:
        # Sampling arguments. This should have the same keys with `operation.mutable_arguments`
        self._sampled: Optional[Dict[str, Any]] = None

    def resample(self, operation: MixedOperation, memo: Dict[str, Any]) -> Dict[str, Any]:
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

    def export(self, operation: MixedOperation, memo: Dict[str, Any]) -> Dict[str, Any]:
        """Export is also random for each leaf value choice."""
        result = {}
        space_spec = operation.search_space_spec()
        for label in space_spec:
            if label not in memo:
                result[label] = random.choice(space_spec[label].values)
        return result

    def forward_argument(self, operation: MixedOperation, name: str) -> Any:
        if self._sampled is None:
            raise ValueError('Need to call resample() before running forward')
        if name in operation.mutable_arguments:
            return self._sampled[name]
        return operation.init_arguments[name]
