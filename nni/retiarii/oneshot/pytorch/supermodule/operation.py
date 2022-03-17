"""
Operations that support weight sharing at a fine-grained level,
which is commonly known as super-kernel, or weight entanglement.

"""

import itertools
from typing import Union, Tuple, Dict, List, Any, Type, Callable, Optional, TypeVar
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.hpo_utils import ParameterSpec
from nni.common.serializer import is_traceable
from nni.retiarii.nn.pytorch.api import ValueChoiceX

from .base import BaseSuperNetModule
from ._valuechoice_utils import traverse_all_options, dedup_inner_choices
from ._operation_utils import Slicable as _S, MaybeWeighted as _W, int_or_int_dict, scalar_or_scalar_dict

T = TypeVar('T')


class MixedOperationSamplingStrategy:
    """
    Algo-related part for mixed Operation.

    :class:`MixedOperation` delegates its resample and export to this strategy (or its subclass),
    so that one Operation can be easily combined with different kinds of sampling.

    One SamplingStrategy corresponds to one mixed operation.
    """

    def __init__(self, operation: 'MixedOperation', memo: Dict[str, Any]) -> None:
        """At init, the sampling strategy can prepare basic parameters,
        and store them in operation if they need back propagation.

        This init is called in :meth:`BaseSuperNetModule.mutate`, after the mixed operation is created.
        So similar to :meth:`BaseSuperNetModule.mutate`,
        memo should also be read and written by the strategy itself.
        """
        pass

    def resample(self, operation: 'MixedOperation', memo: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def export(self, operation: 'MixedOperation', memo: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def forward_argument(self, operation: 'MixedOperation', name: str) -> Any:
        raise NotImplementedError()


class MixedOperation(BaseSuperNetModule):
    """
    TBD

    Utility class for all Operations with ValueChoice as its arguments.

    By design, the mixed op should inherit two super-classes.
    One is this class, which is to control algo-related behavior, such as sampling.
    The other is specific to each Operation, which is to control how the Operation
    interprets the sampling result.

    The class controlling Operation-specific behaviors should have a method called ``init_argument``,
    to customize the behavior when calling ``super().__init__()``. For example::

        def init_argument(self, name, value_choice):
            return max(value_choice.candidates)

    The class should also define a ``bound_type``, to control the matching type in mutate,
    a ``argument_list``, to control which arguments can be dynamically used in ``forward``.
    This list will also be used in mutate for sanity check.
    ``forward``, is to control fprop. The accepted arguments are ``argument_list``,
    appended by forward arguments in the ``bound_type``.
    """

    bound_type: Type[nn.Module]                         # defined in subclass
    argument_list: List[str]                    # defined in subclass

    sampling_strategy: MixedOperationSamplingStrategy

    def init_argument(self, name: str, value_choice: ValueChoiceX) -> Any:
        """Get the initialization argument.
        This is often related to specific operator, rather than algo.
        """
        raise NotImplementedError()

    def forward_with_args(self, *args, **kwargs):
        """To control real fprop. The accepted arguments are ``argument_list``,
        appended by forward arguments in the ``bound_type``."""
        raise NotImplementedError()

    def __init__(self, module_kwargs):
        # Concerned arguments
        self.mutable_arguments: Dict[str, ValueChoiceX] = {}

        # get init default
        init_kwargs = {}

        for key, value in module_kwargs.items():
            if isinstance(value, ValueChoiceX):
                if key not in self.argument_list:
                    raise TypeError(f'Unsupported value choice on argument of {self.bound_type}: {key}')
                init_kwargs[key] = self.init_argument(key, value)
                self.mutable_arguments[key] = value
            else:
                init_kwargs[key] = value

        # get all inner leaf value choices
        self._space_spec: Dict[str, ParameterSpec] = dedup_inner_choices(self.mutable_arguments.values())

        super().__init__(**init_kwargs)

    def resample(self, memo):
        return self.sampling_strategy.resample(self, memo)

    def export(self, memo):
        return self.sampling_strategy.export(self, memo)

    def search_space_spec(self):
        return self._space_spec

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
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
                mixed_op = cls(module.trace_kwargs)

                if 'mixed_op_sampling_strategy' not in mutate_kwargs:
                    raise ValueError('Need to sampling strategy of mixed op, but not found in `mutate_kwargs`.')
                strategy_cls: Type[MixedOperationSamplingStrategy] = mutate_kwargs['mixed_op_sampling_strategy']
                # initialize strategy class
                # this is put in mutate because we need to access memo
                mixed_op.sampling_strategy = strategy_cls(mixed_op, memo)

                return mixed_op

    def forward_argument(self, name: str) -> Any:
        """Get the argument used in forward.
        This if often related to algo. We redirect this to sampling strategy.
        """
        return self.sampling_strategy.forward_argument(self, name)

    def forward(self, *args, **kwargs):
        sampled_args = [self.forward_argument(name) for name in self.argument_list]
        return self.forward_with_args(*sampled_args, *args, **kwargs)


class MixedLinear(MixedOperation, nn.Linear):
    """Mixed linear op. Supported arguments are:

    - ``in_features``
    - ``out_features``

    Prefix of weight and bias will be sliced.
    """

    bound_type = nn.Linear
    argument_list = ['in_features', 'out_features']

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def forward_with_args(self,
                          in_features: int_or_int_dict,
                          out_features: int_or_int_dict,
                          input: torch.Tensor) -> torch.Tensor:

        in_features = _W(in_features)
        out_features = _W(out_features)

        weight = _S(self.weight)[:out_features]
        weight = _S(weight)[:, :in_features]
        if self.bias is None:
            bias = self.bias
        else:
            bias = _S(self.bias)[:out_features]

        return F.linear(input, weight, bias)


_int_or_tuple = Union[int, Tuple[int, int]]


class MixedConv2d(MixedOperation, nn.Conv2d):
    """Mixed conv2d op. Supported arguments are:

    - ``in_channels``
    - ``out_channels``
    - ``groups`` (only supported in path sampling)
    - ``stride`` (only supported in path sampling)
    - ``kernel_size``
    - ``padding`` (only supported in path sampling)
    - ``dilation`` (only supported in path sampling)

    ``padding`` will be the "max" padding in differentiable mode.

    For channels, prefix will be sliced.
    For kernels, we take the small kernel from the center and round it to floor (left top). For example ::

        max_kernel = 5*5, sampled_kernel = 3*3, then we take [1: 4]
        max_kernel = 5*5, sampled_kernel = 2*2, then we take [1: 3]
        □ □ □ □ □   □ □ □ □ □
        □ ■ ■ ■ □   □ ■ ■ □ □
        □ ■ ■ ■ □   □ ■ ■ □ □
        □ ■ ■ ■ □   □ □ □ □ □
        □ □ □ □ □   □ □ □ □ □
    """

    bound_type = nn.Conv2d
    argument_list = [
        'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups'
    ]

    @staticmethod
    def _to_tuple(value: scalar_or_scalar_dict[T]) -> Tuple[T, T]:
        if not isinstance(value, tuple):
            return (value, value)
        return value

    def _to_kernel_slice(self, kernel_size: _int_or_tuple) -> Tuple[slice, slice, slice, slice]:
        # slice the kernel size in the center
        kernel_a, kernel_b = self._to_tuple(kernel_size)

        max_kernel_a, max_kernel_b = self.kernel_size  # self.kernel_size must be a tuple
        kernel_a_left, kernel_b_top = (max_kernel_a - kernel_a) // 2, (max_kernel_b - kernel_b) // 2

        return None, None, slice(kernel_a_left, kernel_a_left + kernel_a), slice(kernel_b_top, kernel_b_top + kernel_b)

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        if name not in ['in_channels', 'out_channels', 'groups', 'stride', 'kernel_size', 'padding', 'dilation']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')

        if name == ['kernel_size', 'padding']:
            all_sizes = set(traverse_all_options(value_choice))
            if any(isinstance(sz, tuple) for sz in all_sizes):
                # maximum kernel should be calculated on every dimension
                return (
                    max(self._to_tuple(sz)[0] for sz in all_sizes),
                    max(self._to_tuple(sz)[1] for sz in all_sizes)
                )
            else:
                return max(all_sizes)

        elif name == 'groups':
            # minimum groups, maximum kernel
            return min(traverse_all_options(value_choice))

        else:
            return max(traverse_all_options(value_choice))

    def forward_with_args(self,
                          in_channels: int_or_int_dict,
                          out_channels: int_or_int_dict,
                          kernel_size: scalar_or_scalar_dict[_int_or_tuple],
                          stride: _int_or_tuple,
                          padding: scalar_or_scalar_dict[_int_or_tuple],
                          dilation: int,
                          groups: int,
                          input: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [stride, dilation, groups]):
            raise ValueError('stride, dilation, groups does not support weighted sampling.')

        in_channels = _W(in_channels)
        out_channels = _W(out_channels)

        # slice prefix
        # For groups > 1, we use groups to slice input weights
        weight = _S(self.weight)[:out_channels]
        weight = _S(weight)[None, :in_channels // groups]

        # slice center
        if isinstance(kernel_size, dict):
            padding = self.padding  # must be a tuple
        kernel_a, kernel_b = self._to_tuple(kernel_size)
        kernel_size = _W(kernel_size)
        max_kernel_a, max_kernel_b = self.kernel_size  # self.kernel_size must be a tuple
        kernel_a_left, kernel_b_top = (max_kernel_a - kernel_a) // 2, (max_kernel_b - kernel_b) // 2
        weight = _S(weight)[None, None, kernel_a_left:kernel_a_left + kernel_a, kernel_b_top:kernel_b_top + kernel_b]

        bias = _S(self.bias)[:out_channels] if self.bias is not None else None

        # The rest parameters only need to be converted to tuple
        stride = self._to_tuple(stride)
        dilation = self._to_tuple(dilation)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, stride, (0, 0), dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class MixedBatchNorm2d(MixedOperation, nn.BatchNorm2d):
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

    bound_type = nn.BatchNorm2d
    argument_list = ['num_features', 'eps', 'momentum']

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def forward_with_args(self,
                          num_features: int_or_int_dict,
                          eps: float,
                          momentum: float,
                          input: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [num_features, eps, momentum]):
            raise ValueError('eps, momentum do not support weighted sampling')

        if isinstance(num_features, dict):
            num_features = self.num_features

        if num_features < self.num_features:
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


class MixedMultiHeadAttention(MixedOperation, nn.MultiheadAttention):
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

    bound_type = nn.MultiheadAttention
    argument_list = ['embed_dim', 'num_heads', 'kdim', 'vdim', 'dropout']

    def _to_proj_slice(self, embed_dim: int,
                       weight_or_bias: Literal['weight', 'bias'] = 'weight') -> _multidim_slice:
        # slice three parts, corresponding to q, k, v respectively
        first_dim = [
            slice(embed_dim),
            slice(self.embed_dim, self.embed_dim + embed_dim),
            slice(self.embed_dim * 2, self.embed_dim * 2 + embed_dim)
        ]
        if weight_or_bias == 'weight':
            return (first_dim, slice(embed_dim))
        else:
            return (first_dim, )

    def forward_with_args(
        self,
        embed_dim: int_or_int_dict, num_heads: int,
        kdim: int_or_int_dict, vdim: int_or_int_dict, dropout: float,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if any(isinstance(arg, dict) for arg in [num_heads, dropout]):
            raise ValueError('num_heads, dropout do not support weighted sampling.')

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if isinstance(embed_dim, dict):
            used_embed_dim = self.embed_dim
        else:
            used_embed_dim = self.embed_dim

        embed_dim = _W(embed_dim)

        # in projection weights & biases has q, k, v weights concatenated together
        in_proj_bias = in_proj_weight = None
        if self.in_proj_bias is not None:
            in_proj_bias = _S(self.in_proj_bias)[self._to_proj_slice(embed_dim)]
        if self.in_proj_weight is not None:
            in_proj_weight = _S(self.in_proj_weight)[self._to_proj_slice(embed_dim), :embed_dim]

        bias_k = _S(self.bias_k)[:, :, :embed_dim] if self.bias_k is not None else None
        bias_v = _S(self.bias_v)[:, :, :embed_dim] if self.bias_v is not None else None
        out_proj_weight = _S(self.out_proj.weight)[:embed_dim, :embed_dim]
        out_proj_bias = _S(self.out_proj.bias)[:embed_dim]

        # The rest part is basically same as pytorch
        if not self._qkv_same_embed_dim:
            kdim = _W(kdim)
            vdim = _W(vdim)

            q_proj = _S(self.q_proj_weight)[:embed_dim, :embed_dim]
            k_proj = _S(self.k_proj_weight)[:embed_dim]
            k_proj = _S(k_proj)[None, :kdim]
            v_proj = _S(self.v_proj_weight)[:embed_dim]
            v_proj = _S(v_proj)[None, :vdim]

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, used_embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj, k_proj_weight=k_proj, v_proj_weight=v_proj)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, used_embed_dim, num_heads,
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


NATIVE_MIXED_OPERATIONS: List[Type[MixedOperation]] = [
    MixedLinear,
    MixedConv2d,
    MixedBatchNorm2d,
    MixedMultiHeadAttention,
]
