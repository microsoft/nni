"""
Operations that support weight sharing at a fine-grained level,
which is commonly known as super-kernel, or weight entanglement.

"""

import itertools
from typing import Union, Tuple, Dict, List, Any, Type, Callable, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.common.hpo_utils import ParameterSpec
from nni.common.serializer import is_traceable
from nni.retiarii.nn.pytorch.api import ValueChoiceX

from .base import BaseSuperNetModule
from .valuechoice_utils import traverse_all_options, dedup_inner_choices


T = TypeVar('T')

_multidim_slice = Tuple[slice, ...]

_scalar_or_scalar_dict = Union[T, Dict[T, float]]
_int_or_int_dict = _scalar_or_scalar_dict[int]


def _slice_weight(weight: torch.Tensor, slice_: Union[_multidim_slice, List[Tuple[_multidim_slice, float]]]) -> torch.Tensor:
    # slice_ can be a tuple of slice, e.g., ([3:6], [2:4])
    # or tuple of slice -> float, e.g. {([3:6],): 0.6, ([2:4],): 0.3}

    if isinstance(slice_, list):
        # for weighted case, we get the corresponding masks. e.g.,
        # {([3:6],): 0.6, ([2:4],): 0.3} => [0, 0, 0.3, 0.9, 0.6, 0.6] (if the whole length is 6)
        # this mask is broadcasted and multiplied onto the weight

        masks = []

        # the accepted argument is list of tuple here
        # because slice can't be key of dict
        for sl, wt in slice_:
            # create a mask with weight w
            with torch.no_grad():
                mask = torch.zeros_like(weight)
                mask[sl] = 1

            # track gradients here
            masks.append((mask * wt))

        masks = sum(masks)
        print(masks)

        return masks * weight

    else:
        # for unweighted case, we slice it directly.

        # sometimes, we don't need slice.
        # this saves an op on computational graph, which will hopefully make training faster
        no_effect = True
        for i in range(len(slice_)):
            s = slice_[i]
            if s is not None and not (
                (s.start is None or s.start == 0) and               # start is useless
                (s.stop is None or s.stop >= weight.size(i)) and    # stop is useless
                s.step in (1, None)                                 # step is useless
            ):
                no_effect = False
        if no_effect:
            return weight

        return weight[slice_]


def _to_slice(
    value: _scalar_or_scalar_dict[T], transform: Callable[[T], slice]
) -> Union[List[Tuple[_multidim_slice, float]], _multidim_slice]:
    # two types of sampled value: a fixed value or a distribution
    # Use transform to transfrom the value to slice respectively
    if isinstance(value, dict):
        return [(transform(v), weight) for v, weight in value.items()]
    else:
        return transform(value)


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


class SuperLinear(MixedOperation, nn.Linear):
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
                          in_features: _int_or_int_dict,
                          out_features: _int_or_int_dict,
                          input: torch.Tensor) -> torch.Tensor:

        in_features = _to_slice(in_features, lambda v: (None, slice(v)))
        out_features = _to_slice(out_features, lambda v: (slice(v),))

        weight = _slice_weight(self.weight, out_features)
        weight = _slice_weight(weight, in_features)
        if self.bias is None:
            bias = self.bias
        else:
            bias = _slice_weight(self.bias, out_features)

        return F.linear(input, weight, bias)


_int_or_tuple = Union[int, Tuple[int, int]]


class SuperConv2d(MixedOperation, nn.Conv2d):
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
    def _to_tuple(value: _int_or_tuple) -> Tuple[int, int]:
        if isinstance(value, int):
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
                          in_channels: _int_or_int_dict,
                          out_channels: _int_or_int_dict,
                          kernel_size: _scalar_or_scalar_dict[_int_or_tuple],
                          stride: _int_or_tuple,
                          padding: _scalar_or_scalar_dict[_int_or_tuple],
                          dilation: int,
                          groups: int,
                          input: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [stride, dilation, groups]):
            raise ValueError('stride, dilation, groups does not support weighted sampling.')

        # slice prefix
        # For groups > 1, we use groups to slice input weights
        in_channels = _to_slice(in_channels, lambda v: (None, slice(v // groups)))
        out_channels = _to_slice(out_channels, lambda v: (slice(v),))

        weight = _slice_weight(self.weight, out_channels)
        weight = _slice_weight(weight, in_channels)

        # slice center
        if isinstance(kernel_size, dict):
            kernel_slice = [(self._to_kernel_slice(ks), wt) for ks, wt in kernel_size.items()]
            # ignore the weighted padding, use maximum padding here
            padding = self.padding  # must be a tuple
        else:
            kernel_slice = self._to_kernel_slice(kernel_size)

        weight = _slice_weight(weight, kernel_slice)

        if self.bias is None:
            bias = self.bias
        else:
            bias = _slice_weight(self.bias, out_channels)

        # The rest parameters only need to be converted to tuple
        stride = self._to_tuple(stride)
        dilation = self._to_tuple(dilation)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, stride, (0, 0), dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class PathSamplingBatchNorm2d(MixedOperation, nn.BatchNorm2d):
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
                          num_features: _int_or_int_dict,
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


class PathSamplingMultiHeadAttention(MixedOperation, nn.MultiheadAttention):
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

    @staticmethod
    def _slice_qkv_weight(src_tensor: torch.Tensor, unit_dim: int, slice_dim: int) -> torch.Tensor:
        if unit_dim == slice_dim:
            return src_tensor
        # slice the parts for q, k, v respectively
        return torch.cat([src_tensor[i * unit_dim: i * unit_dim + slice_dim] for i in range(3)], 0)

    def forward_with_args(
        self,
        embed_dim: _int_or_int_dict, num_heads: _int_or_int_dict,
        kdim: _int_or_int_dict, vdim: _int_or_int_dict, dropout: bool,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

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
