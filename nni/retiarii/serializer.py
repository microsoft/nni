# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings
from typing import Any, TypeVar, Union

from nni.common.serializer import Traceable, is_traceable, trace, _copy_class_wrapper_attributes
from .utils import ModelNamespace

__all__ = ['get_init_parameters_or_fail', 'serialize', 'serialize_cls', 'basic_unit', 'model_wrapper',
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


def basic_unit(cls: T, basic_unit_tag: bool = True) -> Union[T, Traceable]:
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
    _check_wrapped(cls)

    import torch.nn as nn
    assert issubclass(cls, nn.Module), 'When using @basic_unit, the class must be a subclass of nn.Module.'

    cls = trace(cls)
    cls._nni_basic_unit = basic_unit_tag

    # HACK: for torch script
    # https://github.com/pytorch/pytorch/pull/45261
    # https://github.com/pytorch/pytorch/issues/54688
    # I'm not sure whether there will be potential issues
    import torch
    cls._get_nni_attr = torch.jit.ignore(cls._get_nni_attr)
    cls.trace_symbol = torch.jit.unused(cls.trace_symbol)
    cls.trace_args = torch.jit.unused(cls.trace_args)
    cls.trace_kwargs = torch.jit.unused(cls.trace_kwargs)

    return cls


def model_wrapper(cls: T) -> Union[T, Traceable]:
    """
    Wrap the base model (search space). For example,

    .. code-block:: python

        @model_wrapper
        class MyModel(nn.Module):
            ...

    The wrapper serves two purposes:

        1. Capture the init parameters of python class so that it can be re-instantiated in another process.
        2. Reset uid in namespace so that the auto label counting in each model stably starts from zero.

    Currently, NNI might not complain in simple cases where ``@model_wrapper`` is actually not needed.
    But in future, we might enforce ``@model_wrapper`` to be required for base model.
    """
    _check_wrapped(cls)

    import torch.nn as nn
    assert issubclass(cls, nn.Module)

    wrapper = trace(cls)

    class reset_wrapper(wrapper):
        def __init__(self, *args, **kwargs):
            with ModelNamespace():
                super().__init__(*args, **kwargs)

    _copy_class_wrapper_attributes(wrapper, reset_wrapper)
    reset_wrapper.__wrapped__ = wrapper.__wrapped__
    reset_wrapper._nni_model_wrapper = True
    return reset_wrapper


def is_basic_unit(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    return getattr(cls_or_instance, '_nni_basic_unit', False)


def is_model_wrapped(cls_or_instance) -> bool:
    if not inspect.isclass(cls_or_instance):
        cls_or_instance = cls_or_instance.__class__
    return getattr(cls_or_instance, '_nni_model_wrapper', False)


def _check_wrapped(cls: T) -> bool:
    if getattr(cls, '_traced', False) or getattr(cls, '_nni_model_wrapper', False):
        raise TypeError(f'{cls} is already wrapped with trace wrapper (basic_unit / model_wrapper / trace). Cannot wrap again.')
