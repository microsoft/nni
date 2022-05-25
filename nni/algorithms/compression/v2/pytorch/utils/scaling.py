# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from functools import reduce
from typing import List, overload
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
    padding_kernel
        Whether to pad `-1` at the back of kernel_size.
        If set True, for a given tensor when shrinking, padding `-1` until `len(tensor.shape) == len(kernel_size)`;
        for a given expand size when expanding, padding `-1` until `len(expand_size) == len(kernel_size)`.
    """

    def __init__(self, kernel_size: List[int], padding_kernel: bool = False) -> None:
        self.kernel_size = kernel_size
        self.padding_kernel = padding_kernel

    def _padding(self, _list: List[int], length: int, padding_value: int = -1, padding_mode: Literal['front', 'back'] = 'back') -> List[int]:
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

    def _shrink(self, target: Tensor, kernel_size: List[int]) -> Tensor:
        """
        Main logic about how to shrink target. Subclass could override this function to customize.
        Add all values covered by the kernel as a simple implementation.
        """
        # step 1: reduce dimensions of the target tensor where -1 is located in kernel_size.
        reduced_dims = [dim for (dim, step) in enumerate(kernel_size) if step == -1]
        new_target = target.sum(reduced_dims) if len(reduced_dims) > 0 else target

        # step 2: pooling the new target with remaining kernel_size.
        remaining_kernel_size = [step for step in kernel_size if step != -1]
        letter_candidates = 'abcdefghijklmnopqrstuvwxyz'
        ein_expression = ''
        for i, step in enumerate(remaining_kernel_size):
            new_target = new_target.unfold(i, step, step)
            ein_expression += letter_candidates[i]
        ein_expression = '...{},{}'.format(ein_expression, ein_expression)
        result = torch.einsum(ein_expression, new_target, torch.ones(remaining_kernel_size).to(new_target.device))

        return result

    def _expand(self, target: Tensor, kernel_size: List[int], expand_size: List[int]) -> Tensor:
        """
        Main logic about how to expand target to a specific size. Subclass could override this function to customize.
        Repeat each value to reach the kernel size as a simple implementation.
        """
        # step 1: unsqueeze the target tensor where -1 is located in kernel_size.
        unsqueezed_dims = [dim for (dim, step) in enumerate(kernel_size) if step == -1]
        new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim), [target] + unsqueezed_dims)  # type: ignore

        # step 2: build the _expand_size and unsqueeze target tensor on each dim
        _expand_size = []
        for a, b in zip(kernel_size, expand_size):
            if a == -1:
                _expand_size.append(b)
                _expand_size.append(1)
            else:
                assert b % a == 0
                _expand_size.append(b // a)
                _expand_size.append(a)
        new_target: Tensor = reduce(lambda t, dim: t.unsqueeze(dim), [new_target] + [2 * _ + 1 for _ in range(len(expand_size))])  # type: ignore

        # step 3: expanding the new target to _expand_size and reshape to expand_size.
        result = new_target.expand(_expand_size).reshape(expand_size)

        return result

    def shrink(self, target: Tensor) -> Tensor:
        # Canonicalize kernel_size to target size length at first.
        # If padding_kernel is True, padding -1 at the back of `self.kernel_size`.
        # e.g., padding kernel_size [1] to [1, -1, -1] when target size length is 3.
        # If padding_kernel is False, padding 1 at the front of `self.kernel_size`.
        # e.g., padding kernel_size [2, 2] to [1, 2, 2] when target size length is 3.
        if self.padding_kernel:
            kernel_size = self._padding(self.kernel_size, len(target.shape), -1, 'back')
        else:
            kernel_size = self._padding(self.kernel_size, len(target.shape), 1, 'front')
        return self._shrink(target, kernel_size)

    def expand(self, target: Tensor, expand_size: List[int]):
        # Similar with `self.shrink`, canonicalize kernel_size to expand_size length at first.
        if self.padding_kernel:
            kernel_size = self._padding(self.kernel_size, len(expand_size), -1, 'back')
        else:
            kernel_size = self._padding(self.kernel_size, len(expand_size), 1, 'front')
        return self._expand(target, kernel_size, expand_size)

    @overload
    def validate(self, target: List[int]):
        ...

    @overload
    def validate(self, target: Tensor):
        ...

    def validate(self, target: List[int] | Tensor):
        """
        Validate the target tensor can be shape-lossless scaling.
        That means the shape will not change after `shrink` then `expand`.
        """
        target = target if isinstance(target, Tensor) else torch.rand(target)
        if self.expand((self.shrink(target)), list(target.shape)).shape != target.shape:
            raise ValueError(f'The tensor with shape {target.shape}, can not shape-lossless scaling with ' +
                             f'kernel size is {self.kernel_size} and padding_kernel is {self.padding_kernel}.')
