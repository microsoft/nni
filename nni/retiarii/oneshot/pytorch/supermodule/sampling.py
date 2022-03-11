# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import itertools
import random
from typing import Optional, List, Tuple, Union, Type, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.serializer import is_traceable
from nni.common.hpo_utils import ParameterSpec
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii.nn.pytorch.api import ValueChoiceX
from nni.retiarii.oneshot.pytorch.base_lightning import BaseOneShotLightningModule
from nni.retiarii.oneshot.pytorch.supermodule.valuechoice_utils import dedup_inner_choices, evaluate_value_choice_with_dict

from .base import BaseSuperNetModule
from .valuechoice_utils import *
from .operators import SuperConv2dMixin, SuperLinearMixin


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


class PathSamplingInput(BaseSuperNetModule):
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
        return {
            self.label: ParameterSpec(self.label, 'choice', list(range(self.n_candidates)),
                                      (self.label, ), True, size=self.n_candidates, chosen_size=self.n_chosen)
        }

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


class FineGrainedPathSamplingMixin(BaseSuperNetModule):
    """
    Utility class for all operators with ValueChoice as its arguments.

    By design, the mixed op should inherit two super-classes.
    One is this class, which is to control algo-related behavior, such as sampling.
    The other is specific to each operator, which is to control how the operator
    interprets the sampling result.

    The class controlling operator-specific behaviors should have a method called ``init_argument``,
    to customize the behavior when calling ``super().__init__()``. For example::

        def init_argument(self, name, value_choice):
            return max(value_choice.candidates)

    The class should also define a ``bound_type``, to control the matching type in mutate,
    a ``forward_argument_list``, to control which arguments can be dynamically used in ``forward``.
    This list will also be used in mutate for sanity check.
    ``forward``, is to control fprop. The accepted arguments are ``forward_argument_list``,
    appended by forward arguments in the ``bound_type``.
    """

    bound_type: Type[nn.Module]                         # defined in operator mixin
    init_argument: Callable[[str, ValueChoiceX], Any]   # defined in operator mixin
    forward_argument_list: List[str]                    # defined in eperator mixin

    def __init__(self, module_kwargs):
        # Concerned arguments
        self._mutable_arguments: Dict[str, ValueChoiceX] = {}

        # get init default
        init_kwargs = {}

        for key, value in module_kwargs.items():
            if isinstance(value, ValueChoiceX):
                if key not in self.forward_argument_list:
                    raise TypeError(f'Unsupported value choice on argument of {self.bound_type}: {key}')
                init_kwargs[key] = self.init_argument(key, value)
                self._mutable_arguments[key] = value
            else:
                init_kwargs[key] = value

        # Sampling arguments. This should have the same number of keys as `_mutable_arguments`
        self._sampled: Optional[Dict[str, Any]] = None

        # get all inner leaf value choices
        self._space_spec: Dict[str, ParameterSpec] = dedup_inner_choices(self._mutable_arguments.values())

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
            self._sampled[key] = evaluate_value_choice_with_dict(value, result)

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

    def forward(self, *args, **kwargs):
        sampled_args = [self.get_argument(name) for name in self.forward_argument_list]
        return super().forward(*sampled_args, *args, **kwargs)


class PathSamplingConv2d(FineGrainedPathSamplingMixin, SuperConv2dMixin):
    pass


class PathSamplingLinear(FineGrainedPathSamplingMixin, SuperLinearMixin):
    pass


class PathSamplingBatchNorm2d(FineGrainedPathSamplingMixin, nn.BatchNorm2d):
    """
    TBD
    The BatchNorm2d layer to replace original bn2d with valuechoice in its parameter list. It construct the biggest mean and variation
    tensor first, and slice it before every forward according to the sampled value. Supported parameters are listed below:
        num_features : int
        eps : float
        momentum : float

    Momentum is required to be float.
    PyTorch batchnorm supports a case where momentum can be none, which is not supported here.

    Parameters
    ----------
    module : nn.Module
        the module to be replaced
    name : str
        the unique identifier of `module`
    """

    def default_argument(self, name: str, value_choice: ValueChoiceX):
        if name not in ['num_features', 'eps', 'momentum']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')

        return max(value_choice.all_options())

    def forward(self, input):
        # get sampled parameters
        num_features = self.get_argument('num_features')
        eps = self.get_argument('eps')
        momentum = self.get_argument('momentum')

        weight = self.weight[:num_features]
        bias = self.bias[:num_features]
        running_mean = self.running_mean[:num_features]
        running_var = self.running_var[:num_features]

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            momentum,  # originally exponential_average_factor in pytorch code
            eps,
        )


class PathSamplingMultiHeadAttention(FineGrainedPathSamplingMixin, nn.MultiheadAttention):
    """
    TBD
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

    def default_argument(self, name: str, value_choice: ValueChoiceX) -> Any:
        if name not in ['embed_dim', 'num_heads', 'kdim', 'vdim', 'dropout']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')

        return max(value_choice.all_options())

    @staticmethod
    def _slice_qkv_weight(src_tensor: torch.Tensor, unit_dim: int, slice_dim: int) -> torch.Tensor:
        if unit_dim == slice_dim:
            return src_tensor
        # slice the parts for q, k, v respectively
        return torch.cat([src_tensor[i * unit_dim: i * unit_dim + slice_dim] for i in range(3)], 0)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        embed_dim = self.get_argument('embed_dim')
        num_heads = self.get_argument('num_heads')
        dropout = self.get_argument('dropout')

        # in projection weights & biases has q, k, v weights concatenated together
        if self.in_proj_bias is not None:
            in_proj_bias = self._slice_qkv_weight(self.in_proj_bias, self.embed_dim, embed_dim)
        else:
            in_proj_bias = None

        if self.in_proj_weight is not None:
            in_proj_weight = self._slice_qkv_weight(self.in_proj_weight[:, :embed_dim], self.embed_dim, embed_dim)
        else:
            in_proj_weight = None

        bias_k = self.bias_k[:, :, :embed_dim] if self.bias_k is not None else None
        bias_v = self.bias_v[:, :, :embed_dim] if self.bias_v is not None else None
        out_proj_weight = self.out_proj.weight[:embed_dim, :embed_dim]
        out_proj_bias = self.out_proj.bias[:embed_dim]

        # The rest part is basically same as pytorch
        if not self._qkv_same_embed_dim:
            kdim = self.get_argument('kdim')
            vdim = self.get_argument('vdim')

            q_proj = self.q_proj_weight[:embed_dim, :embed_dim]
            k_proj = self.k_proj_weight[:embed_dim, :kdim]
            v_proj = self.v_proj_weight[:embed_dim, :vdim]

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj, k_proj_weight=k_proj, v_proj_weight=v_proj)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
