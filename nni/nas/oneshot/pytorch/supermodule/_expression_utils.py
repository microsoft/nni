# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities to process the value choice compositions,
in the way that is most convenient to one-shot algorithms."""

from __future__ import annotations

import operator
from typing import Any, TypeVar, List, cast, Mapping, Sequence, Optional, Iterable, overload

import numpy as np
import torch

from nni.mutable import MutableExpression, Categorical


Choice = Any

T = TypeVar('T')

__all__ = [
    'expression_expectation',
    'traverse_all_options',
    'weighted_sum',
    'evaluate_constant',
]


def expression_expectation(mutable_expr: MutableExpression[float] | Any, weights: dict[str, list[float]]) -> float:
    """Compute the expectation of a value choice.

    Parameters
    ----------
    mutable_expr
        The value choice to compute expectation.
    weights
        The weights of each leaf node.

    Returns
    -------
    float
        The expectation.
    """
    if not isinstance(mutable_expr, MutableExpression):
        return mutable_expr

    # Optimization: E(a + b) = E(a) + E(b)
    if hasattr(mutable_expr, 'function') and mutable_expr.function == operator.add:
        return sum(expression_expectation(child, weights) for child in mutable_expr.arguments)
    # E(a - b) = E(a) - E(b)
    if hasattr(mutable_expr, 'function') and mutable_expr.function == operator.sub:
        return expression_expectation(mutable_expr.arguments[0], weights) - expression_expectation(mutable_expr.arguments[1], weights)

    all_options = traverse_all_options(mutable_expr, weights)  # [(option, weight), ...]
    options, option_weights = zip(*all_options)  # ([option, ...], [weight, ...])
    return weighted_sum(options, option_weights)


@overload
def traverse_all_options(mutable_expr: MutableExpression[T]) -> list[T]:
    ...


@overload
def traverse_all_options(
    mutable_expr: MutableExpression[T],
    weights: dict[str, Sequence[float]] | dict[str, list[float]] | dict[str, np.ndarray] | dict[str, torch.Tensor]
) -> list[tuple[T, float]]:
    ...


def traverse_all_options(
    mutable_expr: MutableExpression[T],
    weights: dict[str, Sequence[float]] | dict[str, list[float]] | dict[str, np.ndarray] | dict[str, torch.Tensor] | None = None
) -> list[tuple[T, float]] | list[T]:
    """Traverse all possible computation outcome of a value choice.
    If ``weights`` is not None, it will also compute the probability of each possible outcome.

    NOTE: This function is very similar to ``MutableExpression.grid``,
    but it supports specifying weights for each leaf node.

    Parameters
    ----------
    mutable_expr
        The value choice to traverse.
    weights
        If there's a prior on leaf nodes, and we intend to know the (joint) prior on results,
        weights can be provided. The key is label, value are list of float indicating probability.
        Normally, they should sum up to 1, but we will not check them in this function.

    Returns
    -------
    Results will be sorted and duplicates will be eliminated.
    If weights is provided, the return value will be a list of tuple, with option and its weight.
    Otherwise, it will be a list of options.
    """

    # Validation
    simplified = mutable_expr.simplify()
    for label, param in simplified.items():
        if not isinstance(param, Categorical):
            raise TypeError(f'{param!r} is not a categorical distribution')
        if weights is not None:
            if label not in weights:
                raise KeyError(f'{mutable_expr} depends on a weight with key {label}, but not found in {weights}')
            if len(param) != len(weights[label]):
                raise KeyError(f'Expect weights with {label} to be of length {len(param)}, but {len(weights[label])} found')

    # result is a dict from a option to its weight
    result: dict[T, float] = {}

    sample = {}
    for sample_res in mutable_expr.grid(memo=sample):
        probability = 1.
        if weights is not None:
            for label, chosen in sample.items():
                if isinstance(weights[label], dict):
                    # weights[label] is a dict. Choices are used as keys.
                    probability = probability * weights[label][chosen]
                else:
                    # weights[label] is a list. We need to find the index of currently chosen value.
                    chosen_idx = cast(Categorical, simplified[label]).values.index(chosen)
                    if chosen_idx == -1:
                        raise RuntimeError(f'{chosen} is not a valid value for {label}: {simplified[label]!r}')
                    probability = probability * weights[label][chosen_idx]

        if sample_res in result:
            result[sample_res] = result[sample_res] + cast(float, probability)
        else:
            result[sample_res] = cast(float, probability)

    if weights is None:
        return sorted(result.keys())  # type: ignore
    else:
        return sorted(result.items())  # type: ignore


def evaluate_constant(expr: Any) -> Any:
    """Evaluate a value choice expression to a constant. Raise ValueError if it's not a constant."""
    all_options = traverse_all_options(expr)
    if len(all_options) > 1:
        raise ValueError(f'{expr} is not evaluated to a constant. All possible values are: {all_options}')
    res = all_options[0]
    return res


def weighted_sum(items: Sequence[T], weights: Sequence[float | None] = cast(Sequence[Optional[float]], None)) -> T:
    """Return a weighted sum of items.

    Items can be list of tensors, numpy arrays, or nested lists / dicts.

    If ``weights`` is None, this is simply an unweighted sum.
    """

    if weights is None:
        weights = [None] * len(items)

    assert len(items) == len(weights) > 0
    elem = items[0]
    unsupported_msg = 'Unsupported element type in weighted sum: {}. Value is: {}'

    if isinstance(elem, str):
        # Need to check this first. Otherwise it goes into sequence and causes infinite recursion.
        raise TypeError(unsupported_msg.format(type(elem), elem))

    try:
        if isinstance(elem, (torch.Tensor, np.ndarray, float, int, np.number)):
            if weights[0] is None:
                res = elem
            else:
                res = elem * weights[0]
            for it, weight in zip(items[1:], weights[1:]):
                if type(it) != type(elem):
                    raise TypeError(f'Expect type {type(elem)} but found {type(it)}. Can not be summed')

                if weight is None:
                    res = res + it  # type: ignore
                else:
                    res = res + it * weight  # type: ignore
            return cast(T, res)

        if isinstance(elem, Mapping):
            for item in items:
                if not isinstance(item, Mapping):
                    raise TypeError(f'Expect type {type(elem)} but found {type(item)}')
                if set(item) != set(elem):
                    raise KeyError(f'Expect keys {list(elem)} but found {list(item)}')
            return cast(T, {
                key: weighted_sum(cast(List[dict], [cast(Mapping, d)[key] for d in items]), weights) for key in elem
            })
        if isinstance(elem, Sequence):
            for item in items:
                if not isinstance(item, Sequence):
                    raise TypeError(f'Expect type {type(elem)} but found {type(item)}')
                if len(item) != len(elem):
                    raise ValueError(f'Expect length {len(item)} but found {len(elem)}')
            transposed = cast(Iterable[list], zip(*items))  # type: ignore
            return cast(T, [weighted_sum(column, weights) for column in transposed])
    except (TypeError, ValueError, RuntimeError, KeyError):
        raise ValueError(
            'Error when summing items. Value format / shape does not match. See full traceback for details.' +
            ''.join([
                f'\n  {idx}: {_summarize_elem_format(it)}' for idx, it in enumerate(items)
            ])
        )

    # Dealing with all unexpected types.
    raise TypeError(unsupported_msg)


def _summarize_elem_format(elem: Any) -> Any:
    # Get a summary of one elem
    # Helps generate human-readable error messages

    class _repr_object:
        # empty object is only repr
        def __init__(self, representation):
            self.representation = representation

        def __repr__(self):
            return self.representation

    if isinstance(elem, torch.Tensor):
        return _repr_object('torch.Tensor(' + ', '.join(map(str, elem.shape)) + ')')
    if isinstance(elem, np.ndarray):
        return _repr_object('np.array(' + ', '.join(map(str, elem.shape)) + ')')
    if isinstance(elem, Mapping):
        return {key: _summarize_elem_format(value) for key, value in elem.items()}
    if isinstance(elem, Sequence):
        return [_summarize_elem_format(value) for value in elem]

    # fallback to original, for cases like float, int, ...
    return elem
