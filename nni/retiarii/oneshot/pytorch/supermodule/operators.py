"""
Operations that support weight sharing at a fine-grained level,
which is commonly known as super-kernel, or weight entanglement.

"""

from typing import Union, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoiceX

from .base import BaseSuperNetModule
from .valuechoice_utils import traverse_all_options


def _slice_weight(weight: torch.Tensor, slice_: Union[Tuple[slice], Dict[Tuple[slice], int]]) -> torch.Tensor:
    # slice_ can be a tuple of slice, e.g., ([3:6], [2:4])
    # or tuple of slice -> float, e.g. {([3:6],): 0.6, ([2:4],): 0.3}

    if isinstance(slice_, dict):
        # for weighted case, we get the corresponding masks. e.g.,
        # {([3:6],): 0.6, ([2:4],): 0.3} => [0, 0, 0.3, 0.9, 0.6, 0.6] (if the whole length is 6)
        # this mask is broadcasted and multiplied onto the weight

        new_weight = []

        for sl, wt in slice_.items():
            # create a mask with weight w
            with torch.no_grad():
                mask = torch.zeros_like(weight)
                mask[sl] = 1

            new_weight.append((mask * wt) * weight)

        weight = sum(new_weight)

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


class MixedOperationSamplingStrategy:
    """
    Algo-related part for mixed Operation.

    :class:`MixedOperation` delegates its resample and export to this strategy (or its subclass),
    so that one Operation can be easily combined with different kinds of sampling.
    """

    def resample(self, operation: 'MixedOperation', memo: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def export(self, operation: 'MixedOperation', memo: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_argument(self, name: str) -> Any:
        raise NotImplementedError()


class MixedOperation(BaseSuperNetModule):
    """
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
    a ``forward_argument_list``, to control which arguments can be dynamically used in ``forward``.
    This list will also be used in mutate for sanity check.
    ``forward``, is to control fprop. The accepted arguments are ``forward_argument_list``,
    appended by forward arguments in the ``bound_type``.
    """

    bound_type: Type[nn.Module]                         # defined in subclass
    init_argument: Callable[[str, ValueChoiceX], Any]   # defined in subclass
    forward_argument_list: List[str]                    # defined in subclass

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
        return self.sampling_strategy

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
    def mutate(cls, module, name, memo, sampling_strategy=):
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



class SuperLinearMixin(nn.Linear):
    """Mixed linear op. Supported arguments are:

    - ``in_features``
    - ``out_features``

    Prefix of weight and bias will be sliced.
    """

    bound_type = nn.Linear
    forward_argument_list = ['in_features', 'out_features']

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        return max(traverse_all_options(value_choice))

    def forward(self,
                in_features: Union[int, Dict[int, float]],
                out_features: Union[int, Dict[int, float]],
                input: torch.Tensor) -> torch.Tensor:

        if isinstance(in_features, dict):
            in_features = {slice(dim): weight for dim, weight in in_features.items()}
        else:
            in_features = slice(in_features)

        if isinstance(out_features, dict):
            out_features = {slice(dim): weight for dim, weight in out_features.items()}
        else:
            out_features = slice(out_features)

        weight = _slice_weight(self.weight, (out_features, in_features))
        if self.bias is None:
            bias = self.bias
        else:
            bias = _slice_weight(self.bias, (out_features, ))

        return F.linear(input, weight, bias)


_int_or_tuple = Union[int, Tuple[int, int]]


class SuperConv2dMixin(nn.Conv2d):
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
    forward_argument_list = [
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

    def forward(self,
                in_channels: Union[int, Dict[int, float]],
                out_channels: Union[int, Dict[int, float]],
                kernel_size: Union[_int_or_tuple, Dict[_int_or_tuple, float]],
                stride: _int_or_tuple,
                padding: Union[_int_or_tuple, Dict[_int_or_tuple, float]],
                dilation: int,
                groups: int,
                input: torch.Tensor) -> torch.Tensor:

        if groups > 1:
            # We use groups to slice input weights
            if isinstance(in_channels, int):
                in_channels = in_channels // groups
            else:
                in_channels = {ch // groups: wt for ch, wt in in_channels.items()}

        # slice prefix
        if isinstance(in_channels, dict):
            in_channels = {(None, slice(dim)): wt for dim, wt in in_channels.items()}
        else:
            in_channels = (None, slice(in_channels))

        if isinstance(out_channels, dict):
            out_channels = {(slice(dim),): wt for dim, wt in out_channels.items()}
        else:
            out_channels = (slice(out_channels),)

        weight = _slice_weight(self.weight, out_channels)
        weight = _slice_weight(weight, in_channels)

        # slice center
        if isinstance(kernel_size, dict):
            kernel_slice = {self._to_kernel_slice(ks): wt for ks, wt in kernel_size.items()}
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
