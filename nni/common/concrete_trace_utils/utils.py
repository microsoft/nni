from typing import Any, Callable, Type
import functools

import torch

def run_onlyif_instance(cond_type: Type[Any], return_orig: bool = True, return_const: Any = None):
    def helper(fn):
        if return_orig:
            @functools.wraps(fn)
            def wrapper(*args):
                if isinstance(args[-1], cond_type):
                    return fn(*args)
                return args[-1]
            return wrapper
        else:
            @functools.wraps(fn)
            def wrapper(*args):
                if isinstance(args[-1], cond_type):
                    return fn(*args)
                return return_const
            return wrapper
    return helper

def map_aggregate_zip(fn: Callable, arg0, *args) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if (not isinstance(arg0, torch.Size)) and isinstance(arg0, tuple):
        for arg in args:
            assert (not isinstance(arg, torch.Size)) and isinstance(arg, tuple)
            assert len(arg0) == len(arg)
        return tuple(map_aggregate_zip(fn, *sub_args) for sub_args in zip(arg0, *args))
    elif isinstance(arg0, list):
        for arg in args:
            assert isinstance(arg, list)
            assert len(arg0) == len(arg)
        return list(map_aggregate_zip(fn, *sub_args) for sub_args in zip(arg0, *args))
    elif isinstance(arg0, dict):
        keys = set(arg0.keys())
        keys_len = len(keys)
        for arg in args:
            assert isinstance(arg, dict)
            keys.update(arg.keys())
            assert keys_len == len(keys)
        return {k: map_aggregate_zip(fn, arg0[k], *(arg[k] for arg in args)) for k in keys}
    else:
        # assert not isinstance(arg0, slice)
        return fn(arg0, *args)

def map_aggregate(fn: Callable, a) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if (not isinstance(a, torch.Size)) and isinstance(a, tuple):
        t = tuple(map_aggregate(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, '_fields') else type(a)(*t)
    elif isinstance(a, list):
        return list(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return list((k, map_aggregate(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
    else:
        return fn(a)
