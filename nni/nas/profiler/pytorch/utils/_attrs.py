# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Type, TypeVar, Tuple

from torch import nn
from nni.mutable import Mutable, MutableExpression
from nni.nas.nn.pytorch import ParametrizedModule

T = TypeVar('T')
tuple_1_t = Tuple[T]
tuple_2_t = Tuple[T, T]
tuple_3_t = Tuple[T, T, T]

tuple_n_t = {
    1: tuple_1_t,
    2: tuple_2_t,
    3: tuple_3_t,
}


def _getitem(obj: Any, index: int) -> Any:
    if not isinstance(index, int):
        raise TypeError('Index must be an integer.')

    # Get item directly if obj is a list or tuple.
    if isinstance(obj, (list, tuple)):
        return obj[index]

    # If obj is a mutable, wrap it with a getitem mutable expression.
    if isinstance(obj, Mutable):
        return MutableExpression(_getitem, _getitem.__qualname__ + '({}, {})', [obj, index])

    # Else obj is considered as a single item, return it directly.
    return obj


def _getattr(obj: nn.Module, name: str, expected_type: Type | None = None) -> Any:
    # Smart version of getattr().
    # It's used to unify the behavior of nn.XXX and nas_nn.MutableXXX.

    # Get from args if it's a mutable module.
    if isinstance(obj, ParametrizedModule):
        val = obj.args[name]
    # Get from nn.Module.
    else:
        val = getattr(obj, name)

    # Transform type.
    if expected_type is None:
        return val

    if expected_type is tuple_1_t:
        if isinstance(val, tuple):
            return val
        return (_getitem(val, 0),)
    if expected_type is tuple_2_t:
        if isinstance(val, tuple):
            return val
        return (_getitem(val, 0), _getitem(val, 1))
    if expected_type is tuple_3_t:
        if isinstance(val, tuple):
            return val
        return (_getitem(val, 0), _getitem(val, 1), _getitem(val, 2))
    if expected_type is bool:
        if isinstance(val, Mutable):
            return val
        if isinstance(val, (int, float, str, bool)):
            return bool(val)
        return val is not None
    if expected_type is int:
        if isinstance(val, Mutable):
            return val
        if isinstance(val, (int, float, str, bool)):
            return int(val)
        return 1 if val is not None else 0

    raise TypeError(f'Unsupported type: {expected_type}')
