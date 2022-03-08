# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import random
from typing import Optional, List, Tuple, Union, Type, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.serializer import is_traceable
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii.nn.pytorch.api import ValueChoiceX
from nni.retiarii.oneshot.pytorch.base_lightning import BaseOneShotLightningModule

from .base import BaseSuperNetModule, ParameterSpec


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
        self.sampled: Optional[Union[List[str], str]] = None  # sampled can be either a list of indices or an index
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
        return {self.label: ParameterSpec(self.label, 'choice', self.op_names, (self.label, ), True)}

    @classmethod
    def mutate(cls, module, name, memo):
        if isinstance(module, LayerChoice):
            return cls(list(module.named_children()), module.label)

    def forward(self, *args, **kwargs):
        if not self._sampled:
            raise ValueError('At least one path needs to be sampled before fprop.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled

        res = [getattr(self, samp)(*args, **kwargs) for samp in sampled]
        if len(res) == 1:
            return res[0]
        else:
            return sum(res)

    def __len__(self):
        return len(self.op_names)


class PathSamplingInput(nn.Module):
    """
    Mixed input. Take a list of tensor as input, select some of them and return the sum.

    Attributes
    ----------
    _sampled : int or list of int
        Sampled input indices.
    """

    def __init__(self, n_candidates: int, n_chosen: int, label: str):
        super().__init__()
        self.n_candidates = n_candidates
        self.n_chosen = n_chosen
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
        # FIXME: no way to express n choose k currently
        return {self.label: ParameterSpec(self.label, 'choice', list(range(self.n_candidates)), (self.label, ), True)}

    @classmethod
    def mutate(cls, module, name, memo):
        if isinstance(module, InputChoice):
            if module.reduction != 'sum':
                raise ValueError('Only input choice of sum reduction is supported.')
            return cls(module.n_candidates, module.n_chosen, module.label)

    def forward(self, input_tensors):
        if not self._sampled:
            raise ValueError('At least one path needs to be sampled before fprop.')
        if len(input_tensors) != self.n_candidates:
            raise ValueError(f'Expect {self.n_candidates} input tensors, found {len(input_tensors)}.')
        sampled = [self._sampled] if not isinstance(self._sampled, list) else self._sampled
        res = [input_tensors[samp] for samp in sampled]
        if len(res) == 1:
            return res[0]
        else:
            return sum(res)

    def __len__(self):
        return self.n_candidates


class FineGrainedPathSamplingMixin(BaseOneShotLightningModule):
    """
    Utility class for all operators with ValueChoice as its arguments.
    """

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

        # get all inner leaf value choices
        self._space_spec: Dict[str, ParameterSpec] = {}
        for value_choice in self._mutable_arguments.values():
            for choice in value_choice.inner_choices():
                param_spec = ParameterSpec(choice.label, 'choice', choice.candidates, (choice.label, ), True)
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
                return cls(cls.bound_type, module.trace_kwargs)

    def default_argument(self, name: str, value_choice: ValueChoiceX):
        """Subclass override this method to customize init argument of super-op. For Example, ::

            def default_argument(self, name, value_choice):
                return max(value_choice.candidates)
        """
        raise NotImplementedError()



class PathSamplingSuperLinear(FineGrainedPathSamplingMixin, nn.Linear):
    """
    The Linear layer to replace original linear with valuechoices in its parameter list. It construct the biggest weight matrix first,
    and slice it before every forward according to the sampled value. Supported parameters are listed below:
        in_features : int
        out_features : int

    Parameters
    ----------
    module : nn.Module:
        module to be replaced
    name : str
        the unique identifier of `module`
    """

    def default_argument(self, name: str, value_choice: ValueChoiceX):
        if name not in ['in_features', 'out_features', 'bias']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')
        return max(value_choice.all_options())

    def forward(self, input):
        self.name = name
        self.args = module.trace_kwargs

        init_args = dict(self.args)
        # compulsory params
        init_args['in_features'] = self.max_candidate('in_features')
        init_args['out_features'] = self.max_candidate('out_features')

        super().__init__(**init_args)

    def forward(self, x):
        in_dim = self.sampled_candidate('in_features')
        out_dim = self.sampled_candidate('out_features')

        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]

        return F.linear(x, weights, bias)


class ValueChoiceSuperLayer:
    """
    Layer that has at least one valuechoice in it param list. Basic functions such as getting max/min/sampled candidates are
    implemented in this class.

    Attributes
    ----------
    name : str
        the unique identifier of the module it replaced
    args : Dict[str, Any]
        the parameter list of the original module

    Parameters
    ----------
    module : nn.Module:
        module to be replaced
    module_name : str
        the unique identifier of `module`
    """

    def max_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return max(attr.candidates[f'{self.name}_{attr_name}'])
        return attr

    def min_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return min(attr.candidates[f'{self.name}_{attr_name}'])
        return attr

    def sampled_candidate(self, attr_name, default = None):
        attr = self.args.get(attr_name, default)
        if isinstance(attr, ToSample):
            return attr.sampled_candidate(f'{self.name}_{attr_name}')
        return attr


class ENASValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))
        self.n_chosen = 1


class RandomValueChoice(ToSample):
    def __init__(self, value_choice):
        super().__init__(value_choice.label, len(value_choice.candidates))


class PathSamplingSuperLinear(nn.Linear, ValueChoiceSuperLayer):
    """
    The Linear layer to replace original linear with valuechoices in its parameter list. It construct the biggest weight matrix first,
    and slice it before every forward according to the sampled value. Supported parameters are listed below:
        in_features : int
        out_features : int

    Parameters
    ----------
    module : nn.Module:
        module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name) -> None:
        self.name = name
        self.args = module.trace_kwargs

        init_args = dict(self.args)
        # compulsory params
        init_args['in_features'] = self.max_candidate('in_features')
        init_args['out_features'] = self.max_candidate('out_features')

        super().__init__(**init_args)

    def forward(self, x):
        in_dim = self.sampled_candidate('in_features')
        out_dim = self.sampled_candidate('out_features')

        weights = self.weight[:out_dim, :in_dim]
        bias = self.bias[:out_dim]

        return F.linear(x, weights, bias)


class PathSamplingSuperConv2d(nn.Conv2d, ValueChoiceSuperLayer):
    """
    The Conv2d layer to replace original conv2d with valuechoices in its parameter list. It construct the biggest weight matrix first,
    and slice it before every forward according to the sampled value.
    Supported valuechoice parameters are listed below:
        in_channels : int
        out_channels : int
        kernel_size : int, tuple(int)
        stride : int, tuple(int)
        padding : int, tuple(int)
        dilation : int, tuple(int)
        group : int

    Warnings
    ----------
    Users are supposed to make sure that in different valuechoices with the same label, candidates with the same index should match
    each other. For example, the constraint among `kernel_size`, `padding`, `stride` and `dilation` in a convolutional layer should
    be met. Users ought to design candidates carefully to produce a tensor with correct shape for downstream calculation.

    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        self.name = name
        self.args = module.trace_kwargs

        init_args = dict(self.args)
        # compulsorty params
        init_args['in_channels'] = self.max_candidate('in_channels')
        init_args['out_channels'] = self.max_candidate('out_channels')
        # kernel_size may be an int or tuple, we turn it into a tuple for simplicity
        init_args['kernel_size'] = self.max_kernel_size = self.max_kernel_size_candidate()
        if not isinstance(self.max_kernel_size, tuple):
            self.max_kernel_size = (self.max_kernel_size, self.max_kernel_size)

        # optional params
        # stride, padding and dilation are not necessary for init funtion, since `Conv2d`` directly accessed them in `forward`,
        # which means we can set them just before calling Conv2d.forward
        init_args['groups'] = self.min_candidate('groups', 1)

        super().__init__(**init_args)

    def forward(self, input):
        in_chn = self.sampled_candidate('in_channels')
        out_chn = self.sampled_candidate('out_channels')
        kernel_size = self.sampled_candidate('kernel_size')
        sampled_kernel_a, sampled_kernel_b = kernel_size \
            if isinstance(kernel_size, tuple) else kernel_size, kernel_size

        # Users are supposed to make sure that candidates with the same index match each other.
        # No need to figure if the following three attributes are tuples or not, since Conv2d will handel them.
        self.stride = self.sampled_candidate('stride', 1)
        self.padding = self.sampled_candidate('padding', 0)
        self.dilation = self.sampled_candidate('dilation', 1)

        # F.conv2d will handle `groups`, but we still need to slice weight tensor
        self.groups = self.sampled_candidate('groups', 1)

        # take the small kernel from the center and round it to floor(left top)
        # Example:
        #   max_kernel = 5*5, sampled_kernel = 3*3, then we take [1: 4]
        #   max_kernel = 5*5, sampled_kernel = 2*2, then we take [1: 3]
        #   □ □ □ □ □   □ □ □ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ □ □ □ □
        #   □ □ □ □ □   □ □ □ □ □
        max_kernel_a, max_kernel_b = self.max_kernel_size
        kernel_a_left, kernel_b_top = (max_kernel_a - sampled_kernel_a) // 2, (max_kernel_b - sampled_kernel_b) // 2
        weight = self.weight[:out_chn, :in_chn // self.groups,
            kernel_a_left : kernel_a_left + sampled_kernel_a,
            kernel_b_top : kernel_b_top + sampled_kernel_b]
        bias = self.bias[:out_chn] if self.bias is not None else None

        return self._conv_forward(input, weight, bias)

    def max_kernel_size_candidate(self):
        kernel_size = self.args['kernel_size']

        if not isinstance(kernel_size, ToSample):
            return kernel_size

        candidates = kernel_size.candidates[f'{self.name}_kernel_size']
        if not isinstance(candidates[0], tuple):
            return max(candidates)

        maxa, maxb = 0, 0
        for a, b in candidates:
            a = max(a, maxa)
            b = max(b, maxb)
        return maxa, maxb


class PathSamplingSuperBatchNorm2d(nn.BatchNorm2d, ValueChoiceSuperLayer):
    """
    The BatchNorm2d layer to replace original bn2d with valuechoice in its parameter list. It construct the biggest mean and variation
    tensor first, and slice it before every forward according to the sampled value. Supported parameters are listed below:
        num_features : int
        eps : float
        momentum : float

    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        self.name = name
        self.args = module.trace_kwargs

        init_args = dict(self.args)
        # compulsory params
        init_args['num_features'] = self.max_candidate('num_features')

        # optional params
        # the initial values of eps and momentum doesn't matter since they are directly accessed in forward
        # we just take max candidate for simplicity here
        init_args['eps'] = self.max_candidate('eps', 1e-4)
        init_args['momentum'] = self.max_candidate('momentum', .1)

        super().__init__(**init_args)

    def forward(self, input):
        # get sampled parameters
        num_features = self.sampled_candidate('num_features')
        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        running_mean = self.running_mean[:num_features]
        running_var = self.running_var[:num_features]

        self.eps = self.sampled_candidate('eps', 1e-4)
        self.momentum = self.sampled_candidate('momentum', .1)

        # region
        # code below are simply copied from pytorch v1.10.1 source code since directly setting weight or bias is not allowed.
        # please turn to pytorch source code if you have any problem with code below
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        # endregion

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean
            if not self.training or self.track_running_stats
            else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class PathSamplingMultiHeadAttention(nn.MultiheadAttention, ValueChoiceSuperLayer):
    """
    The MultiHeadAttention layer to replace original mhattn with valuechoice in its parameter list. It construct the biggest Q, K,
    V and some other tensors first, and slice it before every forward according to the sampled value. Supported parameters are listed
    below:
        embed_dim : int
        num_heads : float
        kdim :int
        vdim : int
        dropout : float

    Warnings
    ----------
    Users are supposed to make sure that in different valuechoices with the same label, candidates with the same index should match
    each other. For example, the divisibility constraint between `embed_dim` and `num_heads` in a multi-head attention module should
    be met. Users ought to design candidates carefully to prevent the module from breakdown.

    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """
    def __init__(self, module, name):
        self.name = name
        self.args = module.trace_kwargs

        init_args = dict(self.args)
        # compulsory params
        init_args['embed_dim'] = self.max_embed_dim = self.max_candidate('embed_dim')
        init_args['num_heads'] = self.max_candidate('num_heads')

        # optional params
        init_args['kdim'] = self.max_candidate('kdim', self.max_embed_dim)
        init_args['vdim'] = self.max_candidate('vdim', self.max_embed_dim)
        init_args['dropout'] = self.max_candidate('dropout', 0.)

        super().__init__(**init_args)

    def forward(self, query, key, value, key_padding_mask = None, need_weights = True, attn_mask = None):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        embed_dim = self.sampled_candidate('embed_dim')
        num_heads = self.sampled_candidate('num_heads')
        self.dropout = self.sampled_candidate('dropout', 0.)

        in_proj_bias = torch.concat(
            [self.in_proj_bias[:embed_dim],
             self.in_proj_bias[self.max_embed_dim : self.max_embed_dim + embed_dim],
             self.in_proj_bias[2 * self.max_embed_dim : 2 * self.max_embed_dim + embed_dim]], dim = 0) \
                       if self.in_proj_bias is not None else None
        in_proj_weight = torch.concat(
            [self.in_proj_weight[:embed_dim, :embed_dim],
             self.in_proj_weight[self.max_embed_dim : self.max_embed_dim + embed_dim, :embed_dim],
             self.in_proj_weight[2 * self.max_embed_dim : 2 * self.max_embed_dim + embed_dim, :embed_dim]], dim = 0) \
                         if self.in_proj_weight is not None else None
        bias_k = self.bias_k[:, :, :embed_dim] if self.bias_k is not None else None
        bias_v = self.bias_v[:, :, :embed_dim] if self.bias_v is not None else None
        out_proj_weight = self.out_proj.weight[:embed_dim, :embed_dim]
        out_proj_bias = self.out_proj.bias[:embed_dim]

        if self._qkv_same_embed_dim:

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        else:
            kdim = self.sampled_candidate('kdim', embed_dim)
            vdim = self.sampled_candidate('vdim', embed_dim)

            q_proj = self.q_proj_weight[:embed_dim, :embed_dim]
            k_proj = self.k_proj_weight[:embed_dim, :kdim]
            v_proj = self.v_proj_weight[:embed_dim, :vdim]

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj, k_proj_weight=k_proj, v_proj_weight=v_proj)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
