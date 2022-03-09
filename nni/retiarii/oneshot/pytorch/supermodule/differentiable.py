import warnings

from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.hpo_utils import ParameterSpec
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii.nn.pytorch.api import ValueChoiceX
from nni.retiarii.oneshot.pytorch.base_lightning import BaseOneShotLightningModule

from .base import BaseSuperNetModule


class DifferentiableMixedLayer(BaseSuperNetModule):
    """
    TBD
    Mixed layer, in which fprop is decided by exactly one inner layer or sum of multiple (sampled) layers.
    If multiple modules are selected, the result will be summed and returned.

    Differentiable sampling layer requires all operators returning the same shape for one input,
    as all outputs will be weighted summed to get the final output.

    Attributes
    ----------
    _sampled : int or list of str
        Sampled module indices.
    label : str
        Name of the choice.
    """

    def __init__(self, paths: List[Tuple[str, nn.Module]], alpha: torch.Tensor, label: str):
        super().__init__()
        self.op_names = []
        for name, module in paths:
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, 'There has to be at least one op to choose from.'
        self.label = label
        self._alpha = alpha

    def resample(self, memo):
        """Do nothing. Differentiable layer doesn't need resample."""
        return {}

    def export(self, memo):
        """Choose the operator with the maximum logit."""
        if self.label in memo:
            return {}  # nothing new to export
        return self.op_names[torch.argmax(self._alpha).item()]

    def search_space_spec(self):
        return {self.label: ParameterSpec(self.label, 'choice', self.op_names, (self.label, ),
                                          True, size=len(self.op_names))}

    @classmethod
    def mutate(cls, module, name, memo):
        if isinstance(module, LayerChoice):
            size = len(module)
            if module.label in memo:
                alpha = memo[module.label]
                if len(alpha) != size:
                    raise ValueError(f'Architecture parameter size of same label {module.label} conflict: {len(alpha)} vs. {size}')
            else:
                alpha = nn.Parameter(torch.randn(size) * 1E-3)  # this can be reinitialized later
            return cls(list(module.named_children()), alpha, module.label)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self._alpha, -1).view(*alpha_shape), 0)

    def parameters(self, *args, **kwargs):
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        for name, p in super().named_parameters(*args, **kwargs):
            if name == '_alpha':
                continue
            yield name, p


class DifferentiableMixedInput(BaseOneShotLightningModule):
    """
    TBD
    """

    def __init__(self, n_candidates: int, n_chosen: Optional[int], alpha: torch.Tensor, label: str):
        super().__init__()
        self.n_candidates = n_candidates
        if n_chosen is None:
            warnings.warn('Differentiable architecture search does not support choosing multiple inputs. Assuming one.',
                          RuntimeWarning)
            self.n_chosen = 1
        self.n_chosen = n_chosen
        self.label = label

        self._alpha = alpha

    def resample(self, memo):
        """Do nothing. Differentiable layer doesn't need resample."""
        return {}

    def export(self, memo):
        """Choose the operator with the top logits."""
        if self.label in memo:
            return {}  # nothing new to export
        chosen = sorted(torch.argsort(-self._alpha).cpu().numpy().tolist()[:self.n_chosen])
        if len(chosen) == 1:
            chosen = chosen[0]
        return {self.label: chosen}

    def search_space_spec(self):
        return {
            self.label: ParameterSpec(self.label, 'choice', list(range(self.n_candidates)),
                                      (self.label, ), True, size=self.n_candidates, chosen_size=self.n_chosen)
        }

    @classmethod
    def mutate(cls, module, name, memo):
        if isinstance(module, InputChoice):
            if module.reduction != 'sum':
                raise ValueError('Only input choice of sum reduction is supported.')
            size = module.n_candidates
            if module.label in memo:
                alpha = memo[module.label]
                if len(alpha) != size:
                    raise ValueError(f'Architecture parameter size of same label {module.label} conflict: {len(alpha)} vs. {size}')
            else:
                alpha = nn.Parameter(torch.randn(size) * 1E-3)  # this can be reinitialized later
            return cls(module.n_candidates, module.n_chosen, alpha, module.label)

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self._alpha, -1).view(*alpha_shape), 0)

    def parameters(self, *args, **kwargs):
        for _, p in self.named_parameters(*args, **kwargs):
            yield p

    def named_parameters(self, *args, **kwargs):
        for name, p in super().named_parameters(*args, **kwargs):
            if name == '_alpha':
                continue
            yield name, p


class FineGrainedDifferentiableMixin(BaseOneShotLightningModule):
    """
    TBD
    Utility class for all operators with ValueChoice as its arguments.
    """

    bound_type: Type[nn.Module]

    def __init__(self, **module_kwargs):
        # Concerned arguments
        self._mutable_arguments: Dict[str, ValueChoiceX] = {}

        # get init default
        init_kwargs = {}

        for key, value in module_kwargs.items():
            if isinstance(value, ValueChoiceX):
                init_kwargs[key] = self.init_argument(key, value)
                self._mutable_arguments[key] = value
            else:
                init_kwargs[key] = value

        # Sampling arguments. This should have the same number of keys as `_mutable_arguments`
        self._sampled: Optional[Dict[str, Any]] = None

        # get all inner leaf value choices
        self._space_spec: Dict[str, ParameterSpec] = {}
        for value_choice in self._mutable_arguments.values():
            for choice in value_choice.inner_choices():
                param_spec = ParameterSpec(choice.label, 'choice', choice.candidates, (choice.label, ), True, size=len(choice.candidates))
                if choice.label in self._space_spec:
                    if param_spec != self._space_spec[choice.label]:
                        raise ValueError('Value choice conflict: same label with different candidates: '
                                         f'{param_spec} vs. {self._space_spec[choice.label]}')
                else:
                    self._space_spec[choice.label] = param_spec

        super().__init__(**init_kwargs)

    def resample(self, memo):
        """Random sample for each leaf value choice."""
        result = {}
        for label in self._space_spec:
            if label in memo:
                result[label] = memo[label]
            else:
                result[label] = random.choice(self._space_spec[label])

        # composits to kwargs
        # example: result = {"exp_ratio": 3}, self._sampled = {"in_channels": 48, "out_channels": 96}
        self._sampled = {}
        for key, value in self._mutable_arguments.items():
            choice_inner_values = []
            for choice in value.inner_choices():
                choice_inner_values.append(result[choice.label])
            self._sampled[key] = value.evaluate(choice_inner_values)
        self._sampled = result

        return result

    def export(self, memo):
        """Export is also random for each leaf value choice."""
        result = {}
        for label in self._space_spec:
            if label not in memo:
                result[label] = random.choice(self._space_spec[label])
        return result

    def search_space_spec(self):
        return self._space_spec

    @classmethod
    def mutate(cls, module, name, memo):
        if isinstance(module, cls.bound_type) and is_traceable(module):
            # has valuechoice or not
            has_valuechoice = False
            for arg in itertools.chain(module.trace_args, module.trace_kwargs.values()):
                if isinstance(arg, ValueChoiceX):
                    has_valuechoice = True

            if has_valuechoice:
                if module.trace_args:
                    raise ValueError('ValueChoice on class arguments cannot appear together with ``trace_args``. '
                                     'Please enable ``kw_only`` on nni.trace.')

                # save type and kwargs
                return cls(**module.trace_kwargs)

    def get_argument(self, name: str) -> Any:
        if name in self._mutable_arguments:
            return self._sampled[name]
        return getattr(self, name)

    def default_argument(self, name: str, value_choice: ValueChoiceX):
        """Subclass override this method to customize init argument of super-op. For Example, ::

            def default_argument(self, name, value_choice):
                return max(value_choice.candidates)
        """
        raise NotImplementedError()
