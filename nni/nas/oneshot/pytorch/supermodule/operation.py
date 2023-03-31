# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Operations that support weight sharing at a fine-grained level,
which is commonly known as super-kernel (as in channel search), or weight entanglement.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Type, TypeVar, cast, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nni.mutable import MutableExpression
from nni.nas.nn.pytorch import (
    ParametrizedModule,
    MutableConv2d, MutableLinear, MutableBatchNorm2d, MutableLayerNorm, MutableMultiheadAttention
)

from .base import BaseSuperNetModule
from ._expression_utils import traverse_all_options, evaluate_constant
from ._operation_utils import Slicable as _S, MaybeWeighted as _W, int_or_int_dict, scalar_or_scalar_dict

T = TypeVar('T')

__all__ = [
    'MixedOperationSamplingPolicy',
    'MixedOperation',
    'MixedLinear',
    'MixedConv2d',
    'MixedBatchNorm2d',
    'MixedLayerNorm',
    'MixedMultiHeadAttention',
    'NATIVE_MIXED_OPERATIONS',
]

_diff_not_compatible_error = 'To be compatible with differentiable one-shot strategy, {} in {} must not be mutable.'


class MixedOperationSamplingPolicy:
    """
    Algo-related part for mixed Operation.

    :class:`MixedOperation` delegates its resample and export to this policy (or its subclass),
    so that one Operation can be easily combined with different kinds of sampling.

    One SamplingStrategy corresponds to one mixed operation.
    """

    def __init__(self, operation: 'MixedOperation', memo: dict[str, Any], mutate_kwargs: dict[str, Any]) -> None:
        """At init, the sampling policy can prepare basic parameters,
        and store them in operation if they need back propagation.

        This init is called in :meth:`BaseSuperNetModule.mutate`, after the mixed operation is created.
        So similar to :meth:`BaseSuperNetModule.mutate`,
        memo should also be managed (read and written) by the policy itself.
        """

    def resample(self, operation: 'MixedOperation', memo: dict[str, Any]) -> dict[str, Any]:
        """The handler of :meth:`MixedOperation.resample`."""
        raise NotImplementedError()

    def export(self, operation: 'MixedOperation', memo: dict[str, Any]) -> dict[str, Any]:
        """The handler of :meth:`MixedOperation.export`."""
        raise NotImplementedError()

    def export_probs(self, operation: 'MixedOperation', memo: dict[str, Any]) -> dict[str, Any]:
        """The handler of :meth:`MixedOperation.export_probs`."""
        raise NotImplementedError()

    def forward_argument(self, operation: 'MixedOperation', name: str) -> Any:
        """Computing the argument with ``name`` used in operation's forward.
        Usually a value, or a distribution of value.
        """
        raise NotImplementedError()


class MixedOperation(BaseSuperNetModule):
    """This is the base class for all mixed operations.
    It's what you should inherit to support a new operation with mutable.

    It contains commonly used utilities that will ease the effort to write customized mixed operations,
    i.e., operations with mutable in its arguments.
    To customize, please write your own mixed operation, and add the hook into ``mutation_hooks`` parameter when using the strategy.

    By design, for a mixed operation to work in a specific algorithm,
    at least two classes are needed.

    1. One class needs to inherit this class, to control operation-related behavior,
       such as how to initialize the operation such that the sampled operation can be its sub-operation.
    2. The other one needs to inherit :class:`MixedOperationSamplingPolicy`,
       which controls algo-related behavior, such as sampling.

    The two classes are linked with ``sampling_policy`` attribute in :class:`MixedOperation`,
    whose type is set via ``mixed_op_sampling`` in ``mutate_kwargs`` when
    :meth:`MixedOperation.mutate` is called.

    With this design, one mixed-operation (e.g., MixedConv2d) can work in multiple algorithms
    (e.g., both DARTS and ENAS), saving the engineering effort to rewrite all operations for
    each specific algo.

    This class should also define a ``bound_type``, to control the matching type in mutate,
    an ``argument_list``, to control which arguments can be dynamically used in ``forward``.
    This list will also be used in mutate for sanity check.
    """

    bound_type: Type[nn.Module]                 # defined in subclass
    argument_list: list[str]                    # defined in subclass

    sampling_policy: MixedOperationSamplingPolicy

    def super_init_argument(self, name: str, value_choice: MutableExpression) -> Any:
        """Get the initialization argument when constructing super-kernel, i.e., calling ``super().__init__()``.
        This is often related to specific operator, rather than algo.

        For example::

            def super_init_argument(self, name, value_choice):
                return max(value_choice.grid())
        """
        raise NotImplementedError()

    def __post_init__(self) -> None:
        """Can be used to validate, or to do extra processing after calling ``__init__``."""

    def forward_with_args(self, *args, **kwargs):
        """To control real fprop. The accepted arguments are ``argument_list``,
        appended by forward arguments in the ``bound_type``."""
        raise NotImplementedError()

    def __init__(self, module_kwargs: dict[str, Any]) -> None:
        # Concerned arguments
        self.mutable_arguments: dict[str, MutableExpression] = {}
        # Useful when retrieving arguments without mutable
        self.init_arguments: dict[str, Any] = {**module_kwargs}
        self._fill_missing_init_arguments()

        # get init default
        super_init_kwargs = {}

        for key, value in module_kwargs.items():
            if isinstance(value, MutableExpression):
                if key not in self.argument_list:
                    raise TypeError(f'Unsupported mutable argument of "{self.bound_type}": {key}')
                super_init_kwargs[key] = self.super_init_argument(key, value)
                self.mutable_arguments[key] = value
            else:
                super_init_kwargs[key] = value

        super().__init__(**super_init_kwargs)

        for mutable in self.mutable_arguments.values():
            self.add_mutable(mutable)

        self.__post_init__()

    def freeze(self, sample) -> Any:
        """Freeze the mixed operation to a specific operation.
        Weights will be copied from the mixed operation to the frozen operation.

        The returned operation will be of the ``bound_type``.
        """
        arguments = {**self.init_arguments}
        for name, mutable in self.mutable_arguments.items():
            arguments[name] = mutable.freeze(sample)
        operation = self.bound_type(**arguments)

        # copy weights
        state_dict = self.freeze_weight(**arguments)
        operation.load_state_dict(state_dict)
        return operation

    def resample(self, memo):
        """Delegates to :meth:`MixedOperationSamplingPolicy.resample`."""
        return self.sampling_policy.resample(self, memo)

    def export_probs(self, memo):
        """Delegates to :meth:`MixedOperationSamplingPolicy.export_probs`."""
        return self.sampling_policy.export_probs(self, memo)

    def export(self, memo):
        """Delegates to :meth:`MixedOperationSamplingPolicy.export`."""
        return self.sampling_policy.export(self, memo)

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        """Find value choice in module's arguments and replace the whole module"""
        if isinstance(module, cls.bound_type) and isinstance(module, ParametrizedModule):
            if module.trace_args:
                raise ValueError('Mutable on class arguments cannot appear together with ``trace_args``. '
                                 'Please enable ``kw_only`` on nni.trace.')

            # save type and kwargs
            mixed_op = cls(cast(dict, module.trace_kwargs))

            if 'mixed_op_sampling' not in mutate_kwargs:
                raise ValueError("Need a sampling policy for mixed op, but it's not found in `mutate_kwargs`.")
            policy_cls: Type[MixedOperationSamplingPolicy] = mutate_kwargs['mixed_op_sampling']
            # initialize policy class
            # this is put in mutate because we need to access memo
            mixed_op.sampling_policy = policy_cls(mixed_op, memo, mutate_kwargs)

            return mixed_op

    def forward_argument(self, name: str) -> Any:
        """Get the argument used in forward.
        This if often related to algo. We redirect this to sampling policy.
        """
        return self.sampling_policy.forward_argument(self, name)

    def forward(self, *args, **kwargs):
        """First get sampled arguments, then forward with the sampled arguments (by calling ``forward_with_args``)."""
        sampled_args = [self.forward_argument(name) for name in self.argument_list]
        return self.forward_with_args(*sampled_args, *args, **kwargs)

    def _fill_missing_init_arguments(self) -> None:
        """Set the unspecified init arguments in ``self.init_arguments``.
        For example, in the case of Conv2d, when user didn't specify argument ``stride``,
        this method adds ``stride = 1`` in ``self.init_arguments``.

        This is implemented by inspecting the init signature of ``bound_type``.
        Arguments in complex cases like ``__new__`` or in super-class is not supported.
        """

        def unwrap(cls):
            if not hasattr(cls, '__wrapped__'):
                return cls
            return unwrap(cls.__wrapped__)

        for param in inspect.signature(unwrap(self.bound_type).__init__).parameters.values():
            if param.default is not param.empty and param.name not in self.init_arguments:
                self.init_arguments[param.name] = param.default

    def freeze_weight(self, **kwargs):
        """Slice the params and buffers for subnet forward and state dict.

        The arguments are same as the arguments passed to ``__init__``.
        """
        raise NotImplementedError('freeze_weight is not implemented.')


class MixedLinear(MixedOperation, nn.Linear):
    """Mixed linear operation.

    Supported arguments are:

    - ``in_features``
    - ``out_features``

    Prefix of weight and bias will be sliced.
    """

    bound_type = MutableLinear
    argument_list = ['in_features', 'out_features']

    def super_init_argument(self, name: str, value_choice: MutableExpression):
        return max(value_choice.grid())

    def freeze_weight(self, in_features: int_or_int_dict, out_features: int_or_int_dict, **kwargs) -> Any:
        in_features_ = _W(in_features)
        out_features_ = _W(out_features)

        weight = _S(self.weight)[:out_features_]
        weight = _S(weight)[:, :in_features_]
        bias = self.bias if self.bias is None else _S(self.bias)[:out_features_]

        return {'weight': weight, 'bias': bias}

    def forward_with_args(self,
                          in_features: int_or_int_dict,
                          out_features: int_or_int_dict,
                          inputs: torch.Tensor) -> torch.Tensor:

        params_mapping = self.freeze_weight(in_features, out_features)
        weight, bias = [params_mapping.get(name) for name in ['weight', 'bias']]

        return F.linear(inputs, weight, bias)


_int_or_tuple = Union[int, Tuple[int, int]]


class MixedConv2d(MixedOperation, nn.Conv2d):
    """Mixed conv2d op.

    Supported arguments are:

    - ``in_channels``
    - ``out_channels``
    - ``groups``
    - ``stride`` (only supported in path sampling)
    - ``kernel_size``
    - ``padding``
    - ``dilation`` (only supported in path sampling)

    ``padding`` will be the "max" padding in differentiable mode.

    Mutable ``groups`` is NOT supported in most cases of differentiable mode.
    However, we do support one special case when the group number is proportional to ``in_channels`` and ``out_channels``.
    This is often the case of depth-wise convolutions.

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

    bound_type = MutableConv2d
    argument_list = [
        'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups'
    ]

    @staticmethod
    def _to_tuple(value: scalar_or_scalar_dict[Any]) -> tuple[Any, Any]:
        if not isinstance(value, tuple):
            return (value, value)
        return value

    def super_init_argument(self, name: str, mutable_expr: MutableExpression):
        if name not in ['in_channels', 'out_channels', 'groups', 'stride', 'kernel_size', 'padding', 'dilation']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')

        if name == ['kernel_size', 'padding']:
            all_sizes = set(traverse_all_options(mutable_expr))
            if any(isinstance(sz, tuple) for sz in all_sizes):
                # maximum kernel should be calculated on every dimension
                return (
                    max(self._to_tuple(sz)[0] for sz in all_sizes),
                    max(self._to_tuple(sz)[1] for sz in all_sizes)
                )
            else:
                return max(all_sizes)

        elif name == 'groups':
            if 'in_channels' in self.mutable_arguments:
                # If the ratio is constant, we don't need to try the maximum groups.
                try:
                    constant = evaluate_constant(self.mutable_arguments['in_channels'] / mutable_expr)
                    return max(cast(List[float], traverse_all_options(mutable_expr))) // int(constant)
                except ValueError:
                    warnings.warn(
                        'Both input channels and groups are mutable in a convolution, and their relative ratio is not a constant. '
                        'This can be problematic for most one-shot algorithms. Please check whether this is your intention.',
                        RuntimeWarning
                    )

            # minimum groups, maximum kernel
            return min(traverse_all_options(mutable_expr))

        else:
            return max(traverse_all_options(mutable_expr))

    def freeze_weight(self,
                      in_channels: int_or_int_dict,
                      out_channels: int_or_int_dict,
                      kernel_size: scalar_or_scalar_dict[_int_or_tuple],
                      groups: int_or_int_dict,
                      **kwargs) -> Any:
        rv = self._freeze_weight_impl(in_channels, out_channels, kernel_size, groups)
        rv.pop('in_channels_per_group', None)
        return rv

    def _freeze_weight_impl(self,
                            in_channels: int_or_int_dict,
                            out_channels: int_or_int_dict,
                            kernel_size: scalar_or_scalar_dict[_int_or_tuple],
                            groups: int_or_int_dict,
                            **kwargs) -> Any:
        in_channels_ = _W(in_channels)
        out_channels_ = _W(out_channels)

        # slice prefix
        # For groups > 1, we use groups to slice input weights
        weight = _S(self.weight)[:out_channels_]

        if not isinstance(groups, dict):
            weight = _S(weight)[:, :in_channels_ // groups]
            # palce holder
            in_channels_per_group = None
        else:
            assert 'groups' in self.mutable_arguments
            err_message = 'For differentiable one-shot strategy, when groups is a mutable, ' \
                'in_channels and out_channels should also be a mutable. ' \
                'Also, the ratios of in_channels divided by groups, and out_channels divided by groups ' \
                'should be constants.'
            if 'in_channels' not in self.mutable_arguments or 'out_channels' not in self.mutable_arguments:
                raise ValueError(err_message)
            try:
                in_channels_per_group = evaluate_constant(self.mutable_arguments['in_channels'] / self.mutable_arguments['groups'])
            except ValueError:
                raise ValueError(err_message)
            if in_channels_per_group != int(in_channels_per_group):
                raise ValueError(f'Input channels per group is found to be a non-integer: {in_channels_per_group}')

            # Compute sliced weights and groups (as an integer)
            weight = _S(weight)[:, :int(in_channels_per_group)]

        kernel_a, kernel_b = self._to_tuple(kernel_size)
        kernel_a_, kernel_b_ = _W(kernel_a), _W(kernel_b)
        max_kernel_a, max_kernel_b = self.kernel_size  # self.kernel_size must be a tuple
        kernel_a_left, kernel_b_top = (max_kernel_a - kernel_a_) // 2, (max_kernel_b - kernel_b_) // 2
        weight = _S(weight)[:, :, kernel_a_left:kernel_a_left + kernel_a_, kernel_b_top:kernel_b_top + kernel_b_]

        bias = _S(self.bias)[:out_channels_] if self.bias is not None else None

        return {'weight': weight, 'bias': bias, 'in_channels_per_group': in_channels_per_group}

    def forward_with_args(self,
                          in_channels: int_or_int_dict,
                          out_channels: int_or_int_dict,
                          kernel_size: scalar_or_scalar_dict[_int_or_tuple],
                          stride: _int_or_tuple,
                          padding: scalar_or_scalar_dict[_int_or_tuple],
                          dilation: int,
                          groups: int_or_int_dict,
                          inputs: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [stride, dilation]):
            raise ValueError(_diff_not_compatible_error.format('stride, dilation', 'Conv2d'))

        params_mapping = self._freeze_weight_impl(in_channels, out_channels, kernel_size, groups)
        weight, bias, in_channels_per_group = [
            params_mapping.get(name)
            for name in ['weight', 'bias', 'in_channels_per_group']
        ]

        if isinstance(groups, dict):
            if not isinstance(in_channels_per_group, (int, float)):
                raise ValueError(f'Input channels per group is found to be a non-numberic: {in_channels_per_group}')

            if inputs.size(1) % in_channels_per_group != 0:
                raise RuntimeError(
                    f'Input channels must be divisible by in_channels_per_group, but the input shape is {inputs.size()}, '
                    f'while in_channels_per_group = {in_channels_per_group}'
                )
            else:
                groups = inputs.size(1) // int(in_channels_per_group)

        # slice center
        if isinstance(kernel_size, dict):
            # If kernel size is a dict, ignore choices in padding.
            if isinstance(self.padding, str):
                raise ValueError(f'Use "{self.padding}" in padding is not supported.')
            padding = self.padding  # max padding, must be a tuple

        # The rest parameters only need to be converted to tuple
        stride_ = self._to_tuple(stride)
        dilation_ = self._to_tuple(dilation)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(inputs, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, stride_, (0, 0), dilation_, groups)
        return F.conv2d(inputs, weight, bias, stride_, cast('int | tuple', padding), dilation_, groups)


class MixedBatchNorm2d(MixedOperation, nn.BatchNorm2d):
    """
    Mixed BatchNorm2d operation.

    Supported arguments are:

    - ``num_features``
    - ``eps`` (only supported in path sampling)
    - ``momentum`` (only supported in path sampling)

    For path-sampling, prefix of ``weight``, ``bias``, ``running_mean`` and ``running_var``
    are sliced. For weighted cases, the maximum ``num_features`` is used directly.

    Momentum is required to be float.
    PyTorch BatchNorm supports a case where momentum can be none, which is not supported here.
    """

    bound_type = MutableBatchNorm2d
    argument_list = ['num_features', 'eps', 'momentum']

    def super_init_argument(self, name: str, mutable_expr: MutableExpression):
        return max(traverse_all_options(mutable_expr))

    def freeze_weight(self, num_features: int_or_int_dict, **kwargs) -> Any:
        if isinstance(num_features, dict):
            num_features = self.num_features
        weight, bias = self.weight, self.bias
        running_mean, running_var = self.running_mean, self.running_var

        if num_features < self.num_features:
            weight = weight[:num_features]
            bias = bias[:num_features]
            running_mean = None if running_mean is None else running_mean[:num_features]
            running_var = None if running_var is None else running_var[:num_features]

        return {'weight': weight, 'bias': bias,
                'running_mean': running_mean, 'running_var': running_var}

    def forward_with_args(self,
                          num_features: int_or_int_dict,
                          eps: float,
                          momentum: float,
                          inputs: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [eps, momentum]):
            raise ValueError(_diff_not_compatible_error.format('eps and momentum', 'BatchNorm2d'))

        params_mapping = self.freeze_weight(num_features)
        weight, bias, running_mean, running_var = [
            params_mapping.get(name)
            for name in ['weight', 'bias', 'running_mean', 'running_var']
        ]

        if self.training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)

        return F.batch_norm(
            inputs,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean if not self.training or self.track_running_stats else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            momentum,  # originally exponential_average_factor in pytorch code
            eps,
        )


class MixedLayerNorm(MixedOperation, nn.LayerNorm):
    """
    Mixed LayerNorm operation.

    Supported arguments are:

    - ``normalized_shape``
    - ``eps`` (only supported in path sampling)

    For path-sampling, prefix of ``weight`` and ``bias`` are sliced.
    For weighted cases, the maximum ``normalized_shape`` is used directly.

    eps is required to be float.
    """

    bound_type = MutableLayerNorm
    argument_list = ['normalized_shape', 'eps']

    @staticmethod
    def _to_tuple(value: scalar_or_scalar_dict[Any]) -> tuple[Any, Any]:
        if not isinstance(value, tuple):
            return (value, value)
        return value

    def super_init_argument(self, name: str, mutable_expr: MutableExpression):
        if name not in ['normalized_shape']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')
        all_sizes = set(traverse_all_options(mutable_expr))
        if any(isinstance(sz, (tuple, list)) for sz in all_sizes):
            # transpose
            all_sizes = list(zip(*all_sizes))
            # maximum dim should be calculated on every dimension
            return (max(self._to_tuple(sz)) for sz in all_sizes)
        else:
            return max(all_sizes)

    def freeze_weight(self, normalized_shape, **kwargs) -> Any:
        rv = self._freeze_weight_impl(normalized_shape)
        rv.pop('normalized_shape')
        return rv

    def _freeze_weight_impl(self, normalized_shape, **kwargs) -> Any:
        if isinstance(normalized_shape, dict):
            normalized_shape = self.normalized_shape

        # make it as tuple
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        if isinstance(self.normalized_shape, int):
            normalized_shape = (self.normalized_shape, )

        # slice all the normalized shape
        indices = [slice(0, min(i, j)) for i, j in zip(normalized_shape, self.normalized_shape)]

        # remove _S(*)
        weight = self.weight[indices] if self.weight is not None else None
        bias = self.bias[indices] if self.bias is not None else None

        return {'weight': weight, 'bias': bias, 'normalized_shape': normalized_shape}

    def forward_with_args(self,
                          normalized_shape,
                          eps: float,
                          inputs: torch.Tensor) -> torch.Tensor:

        if any(isinstance(arg, dict) for arg in [eps]):
            raise ValueError(_diff_not_compatible_error.format('eps', 'LayerNorm'))

        params_mapping = self._freeze_weight_impl(normalized_shape)
        weight, bias, normalized_shape = [
            params_mapping.get(name)
            for name in ['weight', 'bias', 'normalized_shape']
        ]

        return F.layer_norm(
            inputs,
            normalized_shape,
            weight,
            bias,
            eps
        )


class MixedMultiHeadAttention(MixedOperation, nn.MultiheadAttention):
    """
    Mixed multi-head attention.

    Supported arguments are:

    - ``embed_dim``
    - ``num_heads`` (only supported in path sampling)
    - ``kdim``
    - ``vdim``
    - ``dropout`` (only supported in path sampling)

    At init, it constructs the largest possible Q, K, V dimension.
    At forward, it slices the prefix to weight matrices according to the sampled value.
    For ``in_proj_bias`` and ``in_proj_weight``, three parts will be sliced and concatenated together:
    ``[0, embed_dim)``, ``[max_embed_dim, max_embed_dim + embed_dim)``,
    ``[max_embed_dim * 2, max_embed_dim * 2 + embed_dim)``.

    Warnings
    ----------
    All candidates of ``embed_dim`` should be divisible by all candidates of ``num_heads``.
    """

    bound_type = MutableMultiheadAttention
    argument_list = ['embed_dim', 'num_heads', 'kdim', 'vdim', 'dropout']

    def __post_init__(self):
        # sometimes super-class believes qkv have the same embed_dim.
        # but actually they do not, because we can have dynamic (mutable) kdim/vdim.

        _qkv_same_embed_dim = True

        for dimension in ['kdim', 'vdim']:
            if self.init_arguments[dimension] is None:
                # must follow embed_dim is this case
                continue

            if getattr(self, dimension) == self.embed_dim and \
                    (dimension in self.mutable_arguments or 'embed_dim' in self.mutable_arguments):
                _qkv_same_embed_dim = False

        if self._qkv_same_embed_dim and not _qkv_same_embed_dim:
            self._qkv_same_embed_dim = _qkv_same_embed_dim

            # adding back missing parameters
            # factory_kwargs could be empty for legacy pytorch versions
            factory_kwargs = {}
            if 'device' in self.init_arguments:
                factory_kwargs['device'] = self.init_arguments['device']
            if 'dtype' in self.init_arguments:
                factory_kwargs['dtype'] = self.init_arguments['dtype']
            self.q_proj_weight = nn.Parameter(torch.empty((self.embed_dim, self.embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((self.embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((self.embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)

            # reset parameters
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

    def super_init_argument(self, name: str, mutable_expr: MutableExpression):
        return max(traverse_all_options(mutable_expr))

    def freeze_weight(self, embed_dim, kdim, vdim, **kwargs):
        rv = self._freeze_weight_impl(embed_dim, kdim, vdim, **kwargs)
        # pop flags and nones, as they won't show in state dict
        rv.pop('qkv_same_embed_dim')
        for k in list(rv):
            if rv[k] is None:
                rv.pop(k)
        return rv

    def _to_proj_slice(self, embed_dim: _W) -> list[slice]:
        # slice three parts, corresponding to q, k, v respectively
        return [
            slice(embed_dim),
            slice(self.embed_dim, self.embed_dim + embed_dim),
            slice(self.embed_dim * 2, self.embed_dim * 2 + embed_dim)
        ]

    def _freeze_weight_impl(self, embed_dim, kdim, vdim, **kwargs):
        # by default, kdim, vdim can be none
        if kdim is None:
            kdim = embed_dim
        if vdim is None:
            vdim = embed_dim

        qkv_same_embed_dim = kdim == embed_dim and vdim == embed_dim

        embed_dim_ = _W(embed_dim)

        # in projection weights & biases has q, k, v weights concatenated together
        in_proj_bias: Tensor | None = None
        in_proj_weight: Tensor | None = None
        if self.in_proj_bias is not None:
            in_proj_bias = _S(cast(Tensor, self.in_proj_bias))[self._to_proj_slice(embed_dim_)]
        if self.in_proj_weight is not None:
            in_proj_weight = _S(cast(Tensor, self.in_proj_weight))[self._to_proj_slice(embed_dim_), :embed_dim_]

        bias_k = _S(cast(Tensor, self.bias_k))[:, :, :embed_dim_] if self.bias_k is not None else None
        bias_v = _S(cast(Tensor, self.bias_v))[:, :, :embed_dim_] if self.bias_v is not None else None
        out_proj_weight = _S(cast(Tensor, self.out_proj.weight))[:embed_dim_, :embed_dim_]
        out_proj_bias = _S(cast(Tensor, self.out_proj.bias))[:embed_dim_] if self.out_proj.bias is not None else None

        if not qkv_same_embed_dim:
            q_proj = _S(cast(Tensor, self.q_proj_weight))[:embed_dim_, :embed_dim_]
            k_proj = _S(cast(Tensor, self.k_proj_weight))[:embed_dim_]
            k_proj = _S(k_proj)[:, :_W(kdim)]
            v_proj = _S(cast(Tensor, self.v_proj_weight))[:embed_dim_]
            v_proj = _S(v_proj)[:, :_W(vdim)]
        else:
            q_proj = k_proj = v_proj = None

        return {
            'in_proj_bias': in_proj_bias, 'in_proj_weight': in_proj_weight,
            'bias_k': bias_k, 'bias_v': bias_v,
            'out_proj.weight': out_proj_weight, 'out_proj.bias': out_proj_bias,
            'q_proj_weight': q_proj, 'k_proj_weight': k_proj, 'v_proj_weight': v_proj,
            'qkv_same_embed_dim': qkv_same_embed_dim
        }

    def forward_with_args(
        self,
        embed_dim: int_or_int_dict, num_heads: int,
        kdim: int_or_int_dict | None, vdim: int_or_int_dict | None,
        dropout: float,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if any(isinstance(arg, dict) for arg in [num_heads, dropout]):
            raise ValueError(_diff_not_compatible_error.format('num_heads and dropout', 'MultiHeadAttention'))

        if getattr(self, 'batch_first', False):
            # for backward compatibility: v1.7 doesn't have batch_first
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if isinstance(embed_dim, dict):
            used_embed_dim = self.embed_dim
        else:
            used_embed_dim = embed_dim

        params_mapping = self._freeze_weight_impl(embed_dim, kdim, vdim)
        in_proj_bias, in_proj_weight, bias_k, bias_v, \
            out_proj_weight, out_proj_bias, q_proj, k_proj, v_proj, qkv_same_embed_dim = [
                params_mapping.get(name)
                for name in ['in_proj_bias', 'in_proj_weight', 'bias_k', 'bias_v',
                             'out_proj.weight', 'out_proj.bias', 'q_proj_weight', 'k_proj_weight',
                             'v_proj_weight', 'qkv_same_embed_dim']
            ]

        # The rest part is basically same as pytorch
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, used_embed_dim, num_heads,
            cast(Tensor, in_proj_weight), cast(Tensor, in_proj_bias),
            bias_k, bias_v, self.add_zero_attn,
            dropout, out_proj_weight, cast(Tensor, out_proj_bias),
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=not qkv_same_embed_dim,
            q_proj_weight=q_proj, k_proj_weight=k_proj, v_proj_weight=v_proj)

        if getattr(self, 'batch_first', False):  # backward compatibility
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


NATIVE_MIXED_OPERATIONS: list[Type[MixedOperation]] = [
    MixedLinear,
    MixedConv2d,
    MixedBatchNorm2d,
    MixedLayerNorm,
    MixedMultiHeadAttention,
]

# For the supported operations to be properly rendered in documentation
NATIVE_SUPPORTED_OP_NAMES: list[str] = [op.bound_type.__name__ for op in NATIVE_MIXED_OPERATIONS]
