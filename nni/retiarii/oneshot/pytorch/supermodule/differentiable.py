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



class DifferentiableSuperConv2d(nn.Conv2d):
    """
    TBD
    Only ``kernel_size`` ``in_channels`` and ``out_channels`` are supported. Kernel size candidates should be larger or smaller
    than each other in both candidates. See examples below:
    the following example is not allowed:
        >>> ValueChoice(candidates = [(5, 3), (3, 5)])
            □ ■ ■ ■ □   □ □ □ □ □
            □ ■ ■ ■ □   ■ ■ ■ ■ ■    # candidates are not bigger or smaller on both dimension
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   ■ ■ ■ ■ ■
            □ ■ ■ ■ □   □ □ □ □ □
    the following 3 examples are valid:
        >>> ValueChoice(candidates = [5, 3, 1])
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ ■ □ □
            ■ ■ ■ ■ ■   □ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
        >>> ValueChoice(candidates = [(5, 7), (3, 5), (1, 3)])
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ ■ ■ ■ □ □
            ■ ■ ■ ■ ■ ■ ■  □ ■ ■ ■ ■ ■ □   □ □ □ □ □ □ □
            ■ ■ ■ ■ ■ ■ ■  □ □ □ □ □ □ □   □ □ □ □ □ □ □
        >>> # when the difference between any two candidates is not even, the left upper will be picked:
        >>> ValueChoice(candidates = [(5, 5), (4, 4), (3, 3)])
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ □ □ □ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   ■ ■ ■ ■ □   □ ■ ■ ■ □
            ■ ■ ■ ■ ■   □ □ □ □ □   □ □ □ □ □
    """

    def __init__(self, module, name):
        self.label = name
        args = module.trace_kwargs

        # compulsory params
        if isinstance(args['in_channels'], ValueChoice):
            args['in_channels'] = max(args['in_channels'].candidates)

        self.out_channel_candidates = None
        if isinstance(args['out_channels'], ValueChoice):
            self.out_channel_candidates = sorted(args['out_channels'].candidates, reverse=True)
            args['out_channels'] = self.out_channel_candidates[0]

        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        self.kernel_size_candidates = None
        if isinstance(args['kernel_size'], ValueChoice):
            # unify kernel size as tuple
            candidates = args['kernel_size'].candidates
            if not isinstance(candidates[0], tuple):
                candidates = [(k, k) for k in candidates]

            # sort kernel size in descending order
            self.kernel_size_candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
            for i in range(0, len(self.kernel_size_candidates) - 1):
                bigger = self.kernel_size_candidates[i]
                smaller = self.kernel_size_candidates[i + 1]
                assert bigger[1] > smaller[1] or (bigger[1] == smaller[1] and bigger[0] > smaller[0]), f'Kernel_size candidates ' \
                    f'should be larger or smaller than each other on both dimensions, but found {bigger} and {smaller}.'
            args['kernel_size'] = self.kernel_size_candidates[0]

        super().__init__(**args)
        self.generate_architecture_params()

    def forward(self, input):
        # Note that there is no need to handle ``in_channels`` here since it is already handle by the ``out_channels`` in the
        # previous module. If we multiply alpha with refer to ``in_channels`` here again, the alpha will indeed be considered
        # twice, which is not what we expect.
        weight = self.weight

        def sum_weight(input_weight, masks, thresholds, indicator):
            """
            This is to get the weighted sum of weight.

            Parameters
            ----------
            input_weight : Tensor
                the weight to be weighted summed
            masks : List[Tensor]
                weight masks.
            thresholds : List[float]
                thresholds, should have a length of ``len(masks) - 1``
            indicator : Callable[[Tensor, float], float]
                take a tensor and a threshold as input, and output the weight

            Returns
            ----------
            weight : Tensor
                weighted sum of ``input_weight``. this is of the same shape as ``input_sum``
            """
            # Note that ``masks`` and ``thresholds`` have different lengths. There alignment is shown below:
            # self.xxx_candidates = [   c_0  ,   c_1  , ... ,  c_n-2  ,   c_n-1 ] # descending order
            # self.xxx_mask       = [ mask_0 , mask_1 , ... , mask_n-2, mask_n-1]
            # self.t_xxx          = [   t_0  ,   t_2  , ... ,  t_n-2 ]
            # So we zip the first n-1 items, and multiply masks[-1] in the end.
            weight = torch.zeros_like(input_weight)
            for mask, t in zip(masks[:-1], thresholds):
                cur_part = input_weight * mask
                alpha = indicator(cur_part, t)
                weight = (weight + cur_part) * alpha
            # we do not consider skip-op here for out_channel/expansion candidates, which means at least the smallest channel
            # candidate is included
            weight += input_weight * masks[-1]

            return weight

        if self.kernel_size_candidates is not None:
            weight = sum_weight(weight, self.kernel_masks, self.t_kernel, self.Lasso_sigmoid)

        if self.out_channel_candidates is not None:
            weight = sum_weight(weight, self.channel_masks, self.t_expansion, self.Lasso_sigmoid)

        output = self._conv_forward(input, weight, self.bias)
        return output

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        """
        result = {
            'kernel_size': i,
            'out_channels': j
        }
        which means the best candidate for an argument is the i-th one if candidates are sorted in descending order
        """
        result = {}
        eps = 1e-5
        with torch.no_grad():
            if self.kernel_size_candidates is not None:
                weight = torch.zeros_like(self.weight)
                # ascending order
                for i in range(len(self.kernel_size_candidates) - 2, -1, -1):
                    mask = self.kernel_masks[i]
                    t = self.t_kernel[i]
                    cur_part = self.weight * mask
                    alpha = self.Lasso_sigmoid(cur_part, t)
                    if alpha <= eps:  # takes the smaller one
                        result['kernel_size'] = self.kernel_size_candidates[i + 1]
                        break
                    weight = (weight + cur_part) * alpha

                if 'kernel_size' not in result:
                    result['kernel_size'] = self.kernel_size_candidates[0]
            else:
                weight = self.weight

            if self.out_channel_candidates is not None:
                for i in range(len(self.out_channel_candidates) - 2, -1, -1):
                    mask = self.channel_masks[i]
                    t = self.t_expansion[i]
                    alpha = self.Lasso_sigmoid(weight * mask, t)
                    if alpha <= eps:
                        result['out_channels'] = self.out_channel_candidates[i + 1]

                if 'out_channels' not in result:
                    result['out_channels'] = self.out_channel_candidates[0]

        return result

    @staticmethod
    def Lasso_sigmoid(matrix, t):
        """
        A trick that can make use of both the value of bool(lasso > t) and the gradient of sigmoid(lasso - t)

        Parameters
        ----------
        matrix : Tensor
            the matrix to calculate lasso norm
        t : float
            the threshold
        """
        lasso = torch.norm(matrix) - t
        indicator = (lasso > 0).float()  # torch.sign(lasso)
        with torch.no_grad():
            #            indicator = indicator / 2 + .5 # realign indicator from (-1, 1) to (0, 1)
            indicator -= F.sigmoid(lasso)
        indicator += F.sigmoid(lasso)
        return indicator

    def generate_architecture_params(self):
        self.alpha = {}
        if self.kernel_size_candidates is not None:
            # kernel size arch params
            self.t_kernel = nn.Parameter(torch.rand(len(self.kernel_size_candidates) - 1))
            self.alpha['kernel_size'] = self.t_kernel
            # kernel size mask
            self.kernel_masks = []
            for i in range(0, len(self.kernel_size_candidates) - 1):
                big_size = self.kernel_size_candidates[i]
                small_size = self.kernel_size_candidates[i + 1]
                mask = torch.zeros_like(self.weight)
                mask[:, :, :big_size[0], :big_size[1]] = 1          # if self.weight.shape = (out, in, 7, 7), big_size = (5, 5) and
                mask[:, :, :small_size[0], :small_size[1]] = 0      # small_size = (3, 3), mask will look like:
                self.kernel_masks.append(mask)  # 0 0 0 0 0 0 0
            mask = torch.zeros_like(self.weight)  # 0 1 1 1 1 1 0
            mask[:, :, :self.kernel_size_candidates[-1][0], :self.kernel_size_candidates[-1][1]] = 1  # 0 1 0 0 0 1 0
            self.kernel_masks.append(mask)  # 0 1 0 0 0 1 0
            #   0 1 0 0 0 1 0
        if self.out_channel_candidates is not None:  # 0 1 1 1 1 1 0
            # out_channel (or expansion) arch params. we do not consider skip-op here, so we            #   0 0 0 0 0 0 0
            # only generate ``len(self.kernel_size_candidates) - 1 `` thresholds
            self.t_expansion = nn.Parameter(torch.rand(len(self.out_channel_candidates) - 1))
            self.alpha['out_channels'] = self.t_expansion
            self.channel_masks = []
            for i in range(0, len(self.out_channel_candidates) - 1):
                big_channel, small_channel = self.out_channel_candidates[i], self.out_channel_candidates[i + 1]
                mask = torch.zeros_like(self.weight)
                mask[:big_channel] = 1
                mask[:small_channel] = 0
                # if self.weight.shape = (32, in, W, H), big_channel = 16 and small_size = 8, mask will look like:
                # 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                self.channel_masks.append(mask)
            mask = torch.zeros_like(self.weight)
            mask[:self.out_channel_candidates[-1]] = 1
            self.channel_masks.append(mask)


class DifferentiableBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, module, name):
        self.label = name
        args = module.trace_kwargs
        if isinstance(args['num_features'], ValueChoice):
            args['num_features'] = max(args['num_features'].candidates)
        super().__init__(**args)

        # no architecture parameter is needed for BatchNorm2d Layers
        self.alpha = nn.Parameter(torch.tensor([]))

    def export(self):
        """
        No need to export ``BatchNorm2d``. Refer to the ``Conv2d`` layer that has the ``ValueChoice`` as ``out_channels``.
        """
        return -1
