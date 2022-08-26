# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities to process the value choice compositions,
in the way that is most convenient to one-shot algorithms."""

from __future__ import annotations

import itertools
from typing import Any, TypeVar, List, cast, Mapping, Sequence, Optional, Iterable

import numpy as np
import torch

from nni.common.hpo_utils import ParameterSpec
from nni.nas.nn.pytorch.choice import ChoiceOf, ValueChoiceX


Choice = Any

T = TypeVar('T')

__all__ = [
    'dedup_inner_choices',
    'evaluate_value_choice_with_dict',
    'traverse_all_options',
    'weighted_sum',
    'evaluate_constant',
]


def dedup_inner_choices(value_choices: list[ValueChoiceX]) -> dict[str, ParameterSpec]:
    """Find all leaf nodes in ``value_choices``,
    save them into in the format of ``{label: parameter_spec}``.
    """
    result = {}
    for value_choice in value_choices:
        for choice in value_choice.inner_choices():
            param_spec = ParameterSpec(choice.label, 'choice', choice.candidates, (choice.label, ), True, size=len(choice.candidates))
            if choice.label in result:
                if param_spec != result[choice.label]:
                    raise ValueError('Value choice conflict: same label with different candidates: '
                                     f'{param_spec} vs. {result[choice.label]}')
            else:
                result[choice.label] = param_spec
    return result


def evaluate_value_choice_with_dict(value_choice: ChoiceOf[T], chosen: dict[str, Choice]) -> T:
    """To evaluate a composition of value-choice with a dict,
    with format of ``{label: chosen_value}``.
    The implementation is two-pass. We first get a list of values,
    then feed the values into ``value_choice.evaluate``.
    This can be potentially optimized in terms of speed.

    Examples
    --------
    >>> chosen = {"exp_ratio": 3}
    >>> evaluate_value_choice_with_dict(value_choice_in, chosen)
    48
    >>> evaluate_value_choice_with_dict(value_choice_out, chosen)
    96
    """
    choice_inner_values = []
    for choice in value_choice.inner_choices():
        if choice.label not in chosen:
            raise KeyError(f'{value_choice} depends on a value with key {choice.label}, but not found in {chosen}')
        choice_inner_values.append(chosen[choice.label])
    return value_choice.evaluate(choice_inner_values)


def traverse_all_options(
    value_choice: ChoiceOf[T],
    weights: dict[str, list[float]] | dict[str, np.ndarray] | dict[str, torch.Tensor] | None = None
) -> list[tuple[T, float]] | list[T]:
    """Traverse all possible computation outcome of a value choice.
    If ``weights`` is not None, it will also compute the probability of each possible outcome.

    Parameters
    ----------
    value_choice : ValueChoiceX
        The value choice to traverse.
    weights : Optional[dict[str, list[float]]], default = None
        If there's a prior on leaf nodes, and we intend to know the (joint) prior on results,
        weights can be provided. The key is label, value are list of float indicating probability.
        Normally, they should sum up to 1, but we will not check them in this function.

    Returns
    -------
    list[Union[tuple[Any, float], Any]]
        Results will be sorted and duplicates will be eliminated.
        If weights is provided, the return value will be a list of tuple, with option and its weight.
        Otherwise, it will be a list of options.
    """
    # get a dict of {label: list of tuple of choice and weight}
    leafs: dict[str, list[tuple[T, float]]] = {}
    for label, param_spec in dedup_inner_choices([value_choice]).items():
        if weights is not None:
            if label not in weights:
                raise KeyError(f'{value_choice} depends on a weight with key {label}, but not found in {weights}')
            if len(weights[label]) != param_spec.size:
                raise KeyError(f'Expect weights with {label} to be of length {param_spec.size}, but {len(weights[label])} found')
            leafs[label] = list(zip(param_spec.values, cast(List[float], weights[label])))
        else:
            # create a dummy weight of zero, in case that weights are not provided.
            leafs[label] = list(zip(param_spec.values, itertools.repeat(0., param_spec.size)))

    # result is a dict from a option to its weight
    result: dict[T, float | None] = {}
    labels, values = list(leafs.keys()), list(leafs.values())

    if not labels:
        raise ValueError(f'There expects at least one leaf value choice in {value_choice}, but nothing found')

    # get all combinations
    for prod_value in itertools.product(*values):
        # For example,
        # prod_value = ((3, 0.1), ("cat", 0.3), ({"in": 5}, 0.5))
        # the first dim is chosen value, second dim is probability
        # chosen = {"ks": 3, "animal": "cat", "linear_args": {"in": 5}}
        # chosen_weight = np.prod([0.1, 0.3, 0.5])
        chosen = {label: value[0] for label, value in zip(labels, prod_value)}

        eval_res = evaluate_value_choice_with_dict(value_choice, chosen)

        if weights is None:
            result[eval_res] = None
        else:
            # we can't use reduce or inplace product here,
            # because weight can sometimes be tensors
            chosen_weight = prod_value[0][1]
            for value in prod_value[1:]:
                if chosen_weight is None:
                    chosen_weight = value[1]
                else:
                    chosen_weight = chosen_weight * value[1]

            if eval_res in result:
                result[eval_res] = result[eval_res] + chosen_weight
            else:
                result[eval_res] = chosen_weight

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


def weighted_sum(items: list[T], weights: Sequence[float | None] = cast(Sequence[Optional[float]], None)) -> T:
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
