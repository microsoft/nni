# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import builtins
import operator
from typing import Any, Callable, Type
import functools

import torch
from torch.fx import Node

# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
_orig_module_getattribute: Callable = torch.nn.Module.__getattribute__

_orig_agfunc_apply: Callable = torch.autograd.function.Function.apply
_orig_torch_assert: Callable = torch._assert

_orig_type: Callable = builtins.type
_orig_isinstance: Callable = builtins.isinstance
_orig_issubclass: Callable = builtins.issubclass
_orig_getattr: Callable = builtins.getattr

_orig_range: Type[Any] = builtins.range
_orig_int: Type[Any] = builtins.int
_orig_bool: Type[Any] = builtins.bool
_orig_tuple: Type[Any] = builtins.tuple
_orig_list: Type[Any] = builtins.list
_orig_set: Type[Any] = builtins.set
_orig_frozenset: Type[Any] = builtins.frozenset
_orig_dict: Type[Any] = builtins.dict
_orig_map: Type[Any] = builtins.map
_orig_zip: Type[Any] = builtins.zip
_orig_enumerate: Type[Any] = builtins.enumerate
_orig_slice: Type[Any] = builtins.slice
_orig_reversed: Type[Any] = builtins.reversed

_orig_torch_size: Type[Any] = torch.Size
_orig_torch_finfo: Type[Any] = torch.finfo

_orig_len: Callable = builtins.len
_orig_not: Callable = operator.not_
_orig_is: Callable = operator.is_
_orig_is_not: Callable = operator.is_not
_orig_contains: Callable = operator.contains
_orig_index: Callable = operator.index

_orig_all: Callable = builtins.all
_orig_min: Callable = builtins.min
_orig_max: Callable = builtins.max

_orig_node_is_impure: Callable = Node.is_impure


def run_onlyif_instance(cond_type: Type[Any], return_orig: bool = True, return_const: Any = None):
    def helper(fn):
        if return_orig:
            @functools.wraps(fn)
            def wrapper_orig(*args):
                if _orig_isinstance(args[-1], cond_type):
                    return fn(*args)
                return args[-1]
            return wrapper_orig
        else:
            @functools.wraps(fn)
            def wrapper_const(*args):
                if _orig_isinstance(args[-1], cond_type):
                    return fn(*args)
                return return_const
            return wrapper_const
    return helper

def map_recursive(fn: Callable, arg) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if _orig_type(arg) != torch.Size and _orig_isinstance(arg, _orig_tuple):
        t = _orig_tuple(map_recursive(fn, elem) for elem in arg)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(arg, '_fields') else _orig_type(arg)(*t)
    elif _orig_isinstance(arg, _orig_list):
        return _orig_list(map_recursive(fn, elem) for elem in arg)
    elif _orig_isinstance(arg, _orig_dict):
        return {k: map_recursive(fn, v) for k, v in arg.items()}
    else:
        return fn(arg)

def map_recursive_zip(fn: Callable, arg0, *args) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if _orig_type(arg0) != torch.Size and _orig_isinstance(arg0, _orig_tuple):
        for arg in args:
            assert (not _orig_isinstance(arg, torch.Size)) and _orig_isinstance(arg, _orig_tuple)
            assert len(arg0) == len(arg)
        return _orig_tuple(map_recursive_zip(fn, *sub_args) for sub_args in _orig_zip(arg0, *args))
    elif _orig_isinstance(arg0, _orig_list):
        for arg in args:
            assert _orig_isinstance(arg, _orig_list)
            assert len(arg0) == len(arg)
        return _orig_list(map_recursive_zip(fn, *sub_args) for sub_args in _orig_zip(arg0, *args))
    elif _orig_isinstance(arg0, _orig_dict):
        keys = _orig_set(arg0.keys())
        for arg in args:
            assert _orig_isinstance(arg, _orig_dict) and len(keys.symmetric_difference(arg.keys())) == 0
        return {k: map_recursive_zip(fn, arg0[k], *(arg[k] for arg in args)) for k in keys}
    else:
        # assert not _orig_isinstance(arg0, slice)
        return fn(arg0, *args)
