"""Thie file handles "slice" commonly used in mixed-operation.

The ``slice_type`` we support here, is "slice" or "list of slice".
The reason is that sometimes (e.g., in multi-head attention),
the tensor slice could be from multiple parts. This type is extensible.
We can support arbitrary masks in future if we need them.

To slice a tensor, we need ``multidim_slice``,
which is simply a tuple consists of ``slice_type``.

Usually in python programs, the variable put into slice's start, stop and step
should be integers (or NoneType).
But in our case, it could also be a dict from integer to float,
representing a distribution of integer. When that happens,
we convert a "slice with some weighted values", to a "weighted slice".
To this end, we track the computation with ``MaybeWeighted``,
and replay the computation with each possible value.
Meanwhile, we record their weights.
Note that ``MaybeWeighted`` is also extensible.
We can support more types of objects on slice in future.

The fixed/weighted slice is fed into ``_slice_weight``,
which interprets the slice and apply it on a tensor.
"""

import itertools
import operator
from typing import Tuple, Union, List, Dict, Callable, Optional, Iterator, TypeVar, Any, Generic

import numpy as np
import torch

T = TypeVar('T')

slice_type = Union[slice, List[slice]]
multidim_slice = Tuple[slice_type, ...]

scalar_or_scalar_dict = Union[T, Dict[T, float]]
int_or_int_dict = scalar_or_scalar_dict[int]

_value_fn_type = Optional[Callable[[int_or_int_dict], int]]


def _slice_weight(weight: T, slice_: Union[multidim_slice, List[Tuple[multidim_slice, float]]]) -> T:
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

                if isinstance(sl, list):
                    # slice is a list, meaning that it's assembled from multiple parts
                    for single in sl:
                        mask[single] = 1
                else:
                    mask[sl] = 1

            # track gradients here
            masks.append((mask * wt))

        masks = sum(masks)

        return masks * weight

    else:
        # for unweighted case, we slice it directly.

        def _do_slice(arr, slice_):
            if all(isinstance(s, slice) or s is None for s in slice_):
                # no concat. numpy/torch built-in slice operation is enough.
                return arr[slice_]

            for i in range(len(slice_)):
                if isinstance(slice_[i], list):
                    # if a list, concatenation of multiple parts
                    parts = [arr[tuple([None] * i + [s])] for s in slice_[i]]
                    arr = np.concatenate(parts, i)
                else:
                    # manually slice the i-th dim
                    arr = arr[tuple([None] * i + [slice_[i]])]

            return arr

        # sometimes, we don't need slice.
        # this saves an op on computational graph, which will hopefully make training faster

        # Use a dummy array to check this. Otherwise it would be too complex.
        dummy_arr = np.zeros(weight.shape, dtype=np.bool)
        no_effect = _do_slice(dummy_arr, slice_).shape == dummy_arr.shape

        if no_effect:
            return weight

        return _do_slice(weight, slice_)


class Slicable(Generic[T]):
    """Wraps the weight so that in can be sliced with a "weighted" slice.

    For example::

        weight = conv2d.weight
        Slicable(weight)[P,  {32: 0.4, 64: 0.6}]
    """

    def __init__(self, weight: T):
        self.weight = weight

    def __getitem__(self, index: multidim_slice) -> T:
        if not isinstance(index, tuple):
            index = (index, )

        # Get the dict value in index's leafs
        # There can be at most one dict
        leaf_dict: Optional[Dict[int, float]] = None
        for d in itertools.chain(i.leaf_dicts() for i in _iterate_over_multidim_slice(index)):
            if isinstance(d, dict):
                if leaf_dict is None:
                    leaf_dict = d
                elif leaf_dict is not d:
                    raise ValueError('There can be at most one distinct dict in leaf values.')

        if leaf_dict is None:
            # in case of simple types with no dict
            res_index = _evaluate_multidim_slice(index)
        else:
            # there is a dict, iterate over dict
            res_index = []
            for val, wt in leaf_dict.items():
                res_index_item = _evaluate_multidim_slice(index, lambda _: val)
                res_index.append((res_index_item, wt))

        return _slice_weight(self.weight, res_index)


class MaybeWeighted:
    """Wrap a value (int or dict), so that the computation on it can be replayed.

    For example::

    >>> a = MaybeWeighted({1: 1., 2: 2.}) + 2
    >>> a.apply(1)
    3
    """

    def __init__(self,
                 value: Optional[int_or_int_dict] = None, *,
                 lhs: Optional['MaybeWeighted'] = None,
                 rhs: Optional['MaybeWeighted'] = None,
                 operation: Optional[Callable[[int, int], int]] = None):
        if value is not None:
            self.value = value
        elif lhs is not None and rhs is not None:
            self.lhs = lhs
            self.rhs = rhs
            self.operation = operation

    def leaf_dicts(self) -> Iterator[Dict[int, float]]:
        if self.value is not None:
            yield self.value
        else:
            yield from self.lhs.leaf_dicts()
            yield from self.rhs.leaf_dicts()

    def evaluate(self, value_fn: _value_fn_type = None) -> int:
        if self.value is not None:
            if value_fn is not None:
                return value_fn(self.value)
            return self.value
        else:
            return self.operation(
                self.lhs.evaluate(value_fn),
                self.rhs.evaluate(value_fn)
            )

    def __add__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.add)

    def __radd__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.add)

    def __sub__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.sub)

    def __rsub__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.sub)

    def __mul__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.mul)

    def __rmul__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.mul)

    def __floordiv__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=self, rhs=other, operation=operator.floordiv)

    def __rfloordiv__(self, other: Any) -> 'MaybeWeighted':
        return MaybeWeighted(lhs=other, rhs=self, operation=operator.floordiv)


def _iterate_over_slice_type(s: slice_type):
    if isinstance(s, list):
        for se in s:
            yield from _iterate_over_slice_type(se)
    else:
        # s must be a "slice" now
        if s.start is not None:
            yield s.start
        if s.stop is not None:
            yield s.stop
        if s.step is not None:
            yield s.step


def _iterate_over_multidim_slice(ms: multidim_slice):
    for s in ms:
        if s is not None:
            yield from _iterate_over_slice_type(s)


def _evaluate_slice_type(s: slice_type, value_fn: _value_fn_type = None):
    if isinstance(s, list):
        return [_evaluate_slice_type(se) for se in s]
    else:
        return slice(
            s.start.evaluate(value_fn) if isinstance(s.start, MaybeWeighted) else s.start,
            s.stop.evaluate(value_fn) if isinstance(s.stop, MaybeWeighted) else s.stop,
            s.step.evaluate(value_fn) if isinstance(s.step, MaybeWeighted) else s.step
        )


def _evaluate_multidim_slice(ms: multidim_slice, value_fn: _value_fn_type = None):
    res = []
    for s in ms:
        if s is not None:
            res.append(_evaluate_slice_type(ms, value_fn))
        else:
            res.append(None)
    return tuple(res)
