# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import os
import warnings
from typing import Any, TypeVar, Type

from nni.common.serializer import is_traceable, is_wrapped_with_trace, trace

__all__ = ['get_init_parameters_or_fail', 'serialize', 'serialize_cls', 'basic_unit',
           'is_basic_unit', 'is_model_wrapped']

T = TypeVar('T')


def get_init_parameters_or_fail(obj: Any):
    if is_traceable(obj):
        return obj.trace_kwargs
    raise ValueError(f'Object {obj} needs to be serializable but `trace_kwargs` is not available. '
                     'If it is a built-in module (like Conv2d), please import it from retiarii.nn. '
                     'If it is a customized module, please to decorate it with @basic_unit. '
                     'For other complex objects (e.g., trainer, optimizer, dataset, dataloader), '
                     'try to use @nni.trace.')


def serialize(cls, *args, **kwargs):
    """
    To create an serializable instance inline without decorator. For example,

    .. code-block:: python

        self.op = serialize(MyCustomOp, hidden_units=128)
    """
    warnings.warn('nni.retiarii.serialize is deprecated and will be removed in future release. ' +
                  'Try to use nni.trace, e.g., nni.trace(torch.optim.Adam)(learning_rate=1e-4) instead.',
                  category=DeprecationWarning)
    return trace(cls)(*args, **kwargs)


def serialize_cls(cls):
    """
    To create an serializable class.
    """
    warnings.warn('nni.retiarii.serialize is deprecated and will be removed in future release. ' +
                  'Try to use nni.trace instead.', category=DeprecationWarning)
    return trace(cls)


def basic_unit(cls: T, basic_unit_tag: bool = True) -> T:
    """
    To wrap a module as a basic unit, is to make it a primitive and stop the engine from digging deeper into it.

    ``basic_unit_tag`` is true by default. If set to false, it will not be explicitly mark as a basic unit, and
    graph parser will continue to parse. Currently, this is to handle a special case in ``nn.Sequential``.

    Although ``basic_unit`` calls ``trace`` in its implementation, it is not for serialization. Rather, it is meant
    to capture the initialization arguments for mutation. Also, graph execution engine will stop digging into the inner
    modules when it reaches a module that is decorated with ``basic_unit``.

    .. code-block:: python

        @basic_unit
        class PrimitiveOp(nn.Module):
            ...
    """

    # Internal flag. See nni.trace
    nni_trace_flag = os.environ.get('NNI_TRACE_FLAG', '')
    if nni_trace_flag.lower() == 'disable':
        return cls

    if _check_wrapped(cls, 'basic_unit'):
        return cls

    import torch.nn as nn
    assert issubclass(cls, nn.Module), 'When using @basic_unit, the class must be a subclass of nn.Module.'  # type: ignore

    cls = trace(cls)
    cls._nni_basic_unit = basic_unit_tag  # type: ignore

    _torchscript_patch(cls)

    return cls


def is_basic_unit(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    return getattr(cls_or_instance, '_nni_basic_unit', False)


def is_model_wrapped(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    return getattr(cls_or_instance, '_nni_model_wrapper', False)


def _check_wrapped(cls: Type, rewrap: str) -> bool:
    wrapped = None
    if is_model_wrapped(cls):
        wrapped = 'model_wrapper'
    elif is_basic_unit(cls):
        wrapped = 'basic_unit'
    elif is_wrapped_with_trace(cls):
        wrapped = 'nni.trace'
    if wrapped:
        if wrapped != rewrap:
            raise TypeError(f'{cls} is already wrapped with {wrapped}. Cannot rewrap with {rewrap}.')
        return True
    return False


def _torchscript_patch(cls) -> None:
    # HACK: for torch script
    # https://github.com/pytorch/pytorch/pull/45261
    # https://github.com/pytorch/pytorch/issues/54688
    # I'm not sure whether there will be potential issues
    import torch
    if hasattr(cls, '_get_nni_attr'):  # could not exist on non-linux
        cls._get_nni_attr = torch.jit.ignore(cls._get_nni_attr)
    if hasattr(cls, 'trace_symbol'):
        # these must all exist or all non-exist
        try:
            cls.trace_symbol = torch.jit.unused(cls.trace_symbol)
            cls.trace_args = torch.jit.unused(cls.trace_args)
            cls.trace_kwargs = torch.jit.unused(cls.trace_kwargs)
            cls.trace_copy = torch.jit.ignore(cls.trace_copy)
        except AttributeError as e:
            if 'property' in str(e):
                raise RuntimeError('Trace on PyTorch module failed. Your PyTorch version might be outdated. '
                                   'Please try to upgrade PyTorch.')
            raise
