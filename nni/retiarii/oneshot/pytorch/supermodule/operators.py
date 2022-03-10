"""
Operators that support weight sharing at a fine-grained level,
which is commonly known as super-kernel, or weight entanglement.

"""

from typing import Union, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch.api import ValueChoiceX

from .valuechoice_utils import traverse_all_options


def _slice_weight(weight: torch.Tensor,
                  slice_: Union[slice, Dict[slice, float],
                                Tuple[Union[slice, Dict[slice, float]]]]) -> torch.Tensor:
    if not isinstance(slice_, tuple):
        slice_ = (slice_,)

    # now slice_ is always a tuple
    # each element can be a slice, e.g., [3:6]
    # or a slice -> float, e.g. {[3:6]: 0.6, [2:4]: 0.3}
    # we first handle weighted case, then simple slice

    # for weighted case, we get the corresponding masks. e.g.,
    # {[3:6]: 0.6, [2:4]: 0.3} => [0, 0, 0.3, 0.9, 0.6, 0.6] (if the whole length is 6)
    # this mask is broadcasted and multiplied onto the weight

    # convert weighted slice to ordinary slice after they are processed
    ordinary_slice: List[slice] = []

    for i in range(len(slice_)):
        if isinstance(slice_[i], dict):
            new_weight = []

            for s, w in slice_[i].items():
                # create a mask with weight w
                with torch.no_grad():
                    size = weight.size(i)
                    mask = torch.zeros(size, dtype=weight.dtype, device=weight.device)
                    mask[s] = 1

                    target_shape = [1] * weight.ndim
                    target_shape[i] = size
                    mask = mask.view(*target_shape)

                new_weight.append((mask * w) * weight)

            weight = sum(new_weight)

            ordinary_slice.append(slice(None))

        else:
            # an ordinary slice
            ordinary_slice.append(slice_[i])

    # sometimes, we don't need slice.
    # this saves an op on computational graph, which will hopefully make training faster
    no_effect = True
    for i in range(len(ordinary_slice)):
        s = ordinary_slice[i]
        if not (
            (s.start is None or s.start == 0) and
            (s.stop is None or s.stop >= weight.size(i)) and
            s.step in (1, None)
        ):
            no_effect = False
    if no_effect:
        return weight

    ordinary_slice = tuple(ordinary_slice)
    return weight[ordinary_slice]


class SuperLinearMixin(nn.Linear):
    """Mixed linear op. Supported arguments are:

    - ``in_features``
    - ``out_features``

    Prefix of weight and bias will be sliced.
    """

    bound_type = nn.Linear

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        if name not in ['in_features', 'out_features']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')
        return max(traverse_all_options(value_choice))

    def forward(self, input: torch.Tensor,
                in_features: Union[int, Dict[int, float]],
                out_features: Union[int, Dict[int, float]]):

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
            bias = _slice_weight(self.bias, out_features)

        return F.linear(input, weight, bias)



class SuperConv2dMixin(nn.Conv2d):
    """Mixed conv2d op. Supported arguments are:

    - ``in_channels``
    - ``out_channels``
    - ``groups`` (only supported in path sampling)
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

    def init_argument(self, name: str, value_choice: ValueChoiceX):
        if name not in ['in_channels', 'out_channels', 'groups', 'kernel_size', 'padding', 'dilation']:
            raise NotImplementedError(f'Unsupported value choice on argument: {name}')

        if name == 'kernel_size':
            def int2tuple(ks):
                if isinstance(ks, int):
                    return (ks, ks)
                return ks

            all_kernel_sizes = set(value_choice.all_options())
            if any(isinstance(ks, tuple) for ks in all_kernel_sizes):
                # maximum kernel should be calculated on every dimension
                return (
                    max(int2tuple(ks)[0] for ks in all_kernel_sizes),
                    max(int2tuple(ks)[1] for ks in all_kernel_sizes)
                )
            else:
                return max(all_kernel_sizes)

        elif name == 'groups':
            # minimum groups, maximum kernel
            return min(value_choice.all_options())

        else:
            return max(value_choice.all_options())

    def forward(self, input: torch.Tensor,
                in_channels: Union[int, Dict[int, float]],
                out_channels: Union[int, Dict[int, float]],
                groups: int,
                kernel_size: Union[int, Tuple[int, int], Dict[Union[int, Tuple[int, int]], float]],
                padding: Union[int, Tuple[int, int], Dict[Union[int, Tuple[int, int]], float]],
                dilation: int) -> torch.Tensor:
        # get sampled in/out channels and kernel size
        in_chn = self.get_argument('in_channels')
        out_chn = self.get_argument('out_channels')
        kernel_size = self.get_argument('kernel_size')

        if isinstance(kernel_size, tuple):
            sampled_kernel_a, sampled_kernel_b = kernel_size
        else:
            sampled_kernel_a = sampled_kernel_b = kernel_size

        # F.conv2d will handle `groups`, but we still need to slice weight tensor
        groups = self.get_argument('groups')

        # take the small kernel from the center and round it to floor(left top)
        # Example:
        #   max_kernel = 5*5, sampled_kernel = 3*3, then we take [1: 4]
        #   max_kernel = 5*5, sampled_kernel = 2*2, then we take [1: 3]
        #   □ □ □ □ □   □ □ □ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ ■ ■ □ □
        #   □ ■ ■ ■ □   □ □ □ □ □
        #   □ □ □ □ □   □ □ □ □ □
        max_kernel_a, max_kernel_b = self.kernel_size
        kernel_a_left, kernel_b_top = (max_kernel_a - sampled_kernel_a) // 2, (max_kernel_b - sampled_kernel_b) // 2

        weight = self.weight[:out_chn,
                             :in_chn // self.groups,
                             kernel_a_left: kernel_a_left + sampled_kernel_a,
                             kernel_b_top: kernel_b_top + sampled_kernel_b]
        if self.bias is not None:
            if out_chn < self.out_channels:
                bias = self.bias[:out_chn]
            else:
                bias = self.bias
        else:
            bias = None

        # Users are supposed to make sure that candidates with the same index match each other.
        # The following three attributes must be tuples, since Conv2d will convert them in init if they are not.
        stride = self.get_argument('stride')
        padding = self.get_argument('padding')
        dilation = self.get_argument('dilation')

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, stride, (0, 0), dilation, groups)
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)