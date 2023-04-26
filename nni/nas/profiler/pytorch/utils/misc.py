# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['concat_name', 'standardize_arguments', 'is_leaf_module', 'profiler_leaf_module', 'argument_in_spec']

from typing import Any, Callable, TypeVar, Type

from torch import nn
from nni.nas.nn.pytorch import ParametrizedModule

ModuleType = TypeVar('ModuleType', bound=Type[nn.Module])


def concat_name(name: str, child_name: str) -> str:
    return f'{name}.{child_name}' if name else child_name


def standardize_arguments(args: tuple | Any, process_fn: Callable | None = None) -> tuple[tuple, dict]:
    """
    Standardize the arguments to standard Python arguments.

    Following the pracitce of ``torch.onnx.export``, it accepts three types of arguments and forms them into
    a tuple of positional arguments and a dictionary of keyword arguments:

    1. a tuple of arguments: ``(x, y, z)``
    2. a tensor: ``torch.Tensor([1])``
    3. a tuple of arguments ending with a dictionary of named arguments: ``(x, {"y": input_y, "z": input_z})``

    Parameters
    ----------
    args
        The arguments to standardize.
    process_fn
        A function to process the arguments.

    Returns
    -------
    The standard arguments, positional arguments and keyword arguments.
    """

    if not isinstance(args, tuple):
        args, kwargs = (args,), {}
    elif not args:
        args, kwargs = (), {}
    elif isinstance(args[-1], dict):
        args, kwargs = args[:-1], args[-1]
    else:
        args, kwargs = args, {}

    from torch.utils._pytree import tree_map

    args = tree_map(process_fn, args) if process_fn else args
    kwargs = tree_map(process_fn, kwargs) if process_fn else kwargs
    return args, kwargs


_leaf_registry = []


def is_leaf_module(mod: nn.Module) -> bool:
    """The default implementation of leaf module detection.

    If you want to add more leaf modules, use :func:`profiler_leaf_module` to register them.

    Note that the interpretation of leaf module is finally decided by the profiler.
    We also make no guarantee that the profiler will not peak into a leaf module.
    This is only a utility function to reduce the effort of writing a profiler.
    """
    if isinstance(mod, ParametrizedModule):
        return True
    if any(isinstance(mod, registered) for registered in _leaf_registry):
        return True
    return (mod.__class__.__module__.startswith('torch.nn')
            and not isinstance(mod, nn.Sequential)
            and not isinstance(mod, nn.ModuleList)
            and not isinstance(mod, nn.ModuleDict)
            )


def profiler_leaf_module(mod: ModuleType) -> ModuleType:
    """Register a module as a leaf module for profiler.

    Examples
    --------
    >>> @profiler_leaf_module
    >>> class MyFancyModule(nn.Module):
    ...     pass
    """
    _leaf_registry.append(mod)
    return mod


def argument_in_spec(fn: Any, arg: str) -> bool:
    from inspect import signature
    return arg in signature(fn).parameters
