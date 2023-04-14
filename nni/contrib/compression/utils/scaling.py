# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from functools import reduce
from typing import Callable, List
from typing_extensions import Literal

import torch
from torch import Tensor


class Scaling:
    """
    In the process of generating masks, a large number of operations like pooling or upsampling are involved.
    This class provides tensor-related scaling functions for a given scaling kernel.

    Similar to the concept of convolutional kernel, the scaling kernel also moves over the tensor and does operations.
    The scaling kernel in this class is defined by two parts, kernel size and scaling function (shrink and expand).

    Parameters
    ----------
    kernel_size
        kernel_size is the scale, which determines how large a range in a tensor should shrink to a value,
        or how large a value in a tensor should expand.
        `-1` can be used to indicate that it is a full step in this dimension,
        and the dimension where -1 is located will be reduced or unsqueezed during scaling.

        Example::

            kernel_size = [2, -1]

            # For a given 2D-tensor with size (4, 3),
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]]

            # shrinking it by shrink function, its size becomes (2,) after shrinking:
            [shrink([[1, 2, 3], [4, 5, 6]]), shrink([[7, 8, 9], [10, 11, 12]])]

            # expanding it by expand function with a given expand size,
            # if the expand function is repeating the values, and the expand size is (4, 6, 2):
            [[[1, 1],
              [1, 1],
              [2, 2],
              [2, 2],
              [3, 3],
              [3, 3]],
                ...
              [9, 9]]]
            # note that the original tensor with size (4, 3) will unsqueeze to size (4, 3, 1) at first
            # for the `-1` in kernel_size, then expand size (4, 3, 1) to size (4, 6, 2).
    kernel_padding_mode
        'front' or 'back', default is 'front'.
        If set 'front', for a given tensor when shrinking,
        padding `1` at front of kernel_size until `len(tensor.shape) == len(kernel_size)`;
        for a given expand size when expanding,
        padding `1` at front of kernel_size until `len(expand_size) == len(kernel_size)`.
        If set 'back', for a given tensor when shrinking,
        padding `-1` at back of kernel_size until `len(tensor.shape) == len(kernel_size)`;
        for a given expand size when expanding,
        padding `-1` at back of kernel_size until `len(expand_size) == len(kernel_size)`.

        The default padding value (1 or -1) can be set by passing ``kernel_padding_val``.
    kernel_padding_val
        If ``kernel_padding_val`` is not None, the padding value in kernel padding will be specifed.
    """

    def __init__(self, kernel_size: List[int], kernel_padding_mode: Literal['front', 'back'] = 'front',
                 kernel_padding_val: int | None = None) -> None:
        self.kernel_size = kernel_size
        err_msg = f"kernel_padding_mode should be one of ['front', 'back'], but get kernel_padding_mode={kernel_padding_mode}."
        assert kernel_padding_mode in ['front', 'back'], err_msg
        self.kernel_padding_mode = kernel_padding_mode
        self.kernel_padding_val = kernel_padding_val if kernel_padding_val else (1 if kernel_padding_mode=='front' else -1)

    def _padding(self, _list: List[int], length: int, padding_value: int = -1,
                 padding_mode: Literal['front', 'back'] = 'back') -> List[int]:
        """
        Padding the `_list` to a specific length with `padding_value`.

        Parameters
        ----------
        _list
            The list of int value to be padding.
        length
            The length to pad to.
        padding_value
            Padding value, should be a int.
        padding_mode
            If `padding_mode` is `'front'`, then the padding applied on the front of the size list.
            If `padding_mode` is `'back'`, then the padding applied on the back of the size list.

        Returns
        -------
        List[int]
            The padded list.
        """
        assert len(_list) <= length
        padding = [padding_value for _ in range(length - len(_list))]
        if padding_mode == 'front':
            new_list = padding + list(_list)
        elif padding_mode == 'back':
            new_list = list(_list) + padding
        else:
            raise ValueError(f'Unsupported padding mode: {padding_mode}.')
        return new_list

    def _shrink(self, target: Tensor, kernel_size: List[int], reduce_func: Callable[[Tensor], Tensor] | None = None,
                keepdim: bool = False) -> Tensor:
        """
        Main logic about how to shrink target. Subclass could override this function to customize.
        Sum all values covered by the kernel as a simple implementation.
        """
        # step 1: put the part covered by the kernel to the end of the converted target.
        # e.g., target size is [10, 20], kernel_size is [2, 4], then new_target size is [5, 5, 8].
        reshape_size = []
        final_size = []
        reduced_dims = []
        for (dim, step) in enumerate(kernel_size):
            if step == -1:
                step = target.shape[dim]
                reduced_dims.insert(0, dim)
            assert target.shape[dim] % step == 0
            reshape_size.append(target.shape[dim] // step)
            final_size.append(target.shape[dim] // step)
            reshape_size.append(step)
        permute_dims = [2 * _ for _ in range(len(kernel_size))] + [2 * _ + 1 for _ in range(len(kernel_size))]
        converted_target = target.reshape(reshape_size).permute(permute_dims).reshape(final_size + [-1])

        # step 2: reduce the converted_target last dim with a certain way, by default is converted_target.mean(-1).
        # `sum` does not take into account the metric scale problem, it is better to use `mean` here.
        result = reduce_func(converted_target) if reduce_func else converted_target.mean(-1)

        if not keepdim:
            # step 3: reduce the dims where kernel_size is -1.
            # e.g., target size is [10, 40], kernel_size is [-1, 4], result size is [1, 10], then reduce result to size [10].
            result = reduce(lambda t, dim: t.squeeze(dim), [result] + reduced_dims)  # type: ignore

        return result

    def _expand(self, target: Tensor, kernel_size: List[int], expand_size: List[int], keepdim: bool = False,
                full_expand: bool = True) -> Tensor:
        """
        Main logic about how to expand target to a specific size. Subclass could override this function to customize.
        Repeat each value to reach the kernel size as a simple implementation.
        """
        if not keepdim:
            # step 1: unsqueeze the target tensor where -1 is located in kernel_size.
            unsqueezed_dims = [dim for (dim, step) in enumerate(kernel_size) if step == -1]
            new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim), [target] + unsqueezed_dims)  # type: ignore
        else:
            new_target = target

        # step 2: build the _expand_size and unsqueeze target tensor on each dim
        expand_size = expand_size if full_expand else [1 if a == -1 else b for a, b in zip(kernel_size, expand_size)]
        _expand_size = []
        for a, b in zip(kernel_size, expand_size):
            if a == -1:
                _expand_size.append(1)
                _expand_size.append(b)
            else:
                assert b % a == 0, f'Can not expand tensor with {target.shape} to {expand_size} with kernel size {kernel_size}.'
                _expand_size.append(b // a)
                _expand_size.append(a)
        new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim),
                                    [new_target] + [2 * _ + 1 for _ in range(len(expand_size))])  # type: ignore

        # step 3: expanding the new target to _expand_size and reshape to expand_size.
        # Note that we can also give an interface for how to expand the tensor, like `reduce_func` in `_shrink`,
        # currently we don't have that need.
        result = new_target.expand(_expand_size).reshape(expand_size).clone()

        return result

    def shrink(self, target: Tensor, reduce_func: Callable[[Tensor], Tensor] | None = None, keepdim: bool = False) -> Tensor:
        # Canonicalize kernel_size to target size length at first.
        # If kernel_padding_mode is 'front', padding 1 at the front of `self.kernel_size`.
        # e.g., padding kernel_size [2, 2] to [1, 2, 2] when target size length is 3.
        # If kernel_padding_mode is 'back', padding -1 at the back of `self.kernel_size`.
        # e.g., padding kernel_size [1] to [1, -1, -1] when target size length is 3.
        if self.kernel_padding_mode == 'front':
            kernel_size = self._padding(self.kernel_size, len(target.shape), self.kernel_padding_val, 'front')
        elif self.kernel_padding_mode == 'back':
            kernel_size = self._padding(self.kernel_size, len(target.shape), self.kernel_padding_val, 'back')
        else:
            raise ValueError(f'Unsupported kernel padding mode: {self.kernel_padding_mode}.')
        return self._shrink(target, kernel_size, reduce_func, keepdim)

    def expand(self, target: Tensor, expand_size: List[int], keepdim: bool = False, full_expand: bool = True):
        # If the target from shrink is keepdim, also need to set keepdim when expand.
        # If full_expand is False, the return tensor dim at where kernel_size[dim] == -1 will be 1.
        # Similar with `self.shrink`, canonicalize kernel_size to expand_size length at first.
        if self.kernel_padding_mode == 'front':
            kernel_size = self._padding(self.kernel_size, len(expand_size), self.kernel_padding_val, 'front')
        elif self.kernel_padding_mode == 'back':
            kernel_size = self._padding(self.kernel_size, len(expand_size), self.kernel_padding_val, 'back')
        else:
            raise ValueError(f'Unsupported kernel padding mode: {self.kernel_padding_mode}.')
        return self._expand(target, kernel_size, expand_size, keepdim, full_expand)

    def validate(self, target: List[int] | Tensor):
        """
        Validate the target tensor can be shape-lossless scaling.
        That means the shape will not change after `shrink` then `expand`.
        """
        target = target if isinstance(target, Tensor) else torch.rand(target)
        if self.expand((self.shrink(target)), list(target.shape)).shape != target.shape:
            raise ValueError(f'The tensor with shape {target.shape}, can not shape-lossless scaling with ' +
                             f'kernel size is {self.kernel_size} and kernel_padding_mode is {self.kernel_padding_mode}.')
