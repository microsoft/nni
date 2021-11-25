# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Any, TypeVar, Union

from nni.common.serializer import Traceable, is_traceable, trace, _copy_class_wrapper_attributes
from .utils import ModelNamespace

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
                  'Try to use nni.trace, e.g., nni.trace(torch.optim.Adam)(learning_rate=1e-4) instead.')
    return trace(cls)(*args, **kwargs)


def serialize_cls(cls):
    """
    To create an serializable class.
    """
    warnings.warn('nni.retiarii.serialize is deprecated and will be removed in future release. ' +
                  'Try to use nni.trace instead.')
    return trace(cls)


def basic_unit(cls: T, basic_unit_tag: bool = True) -> Union[T, Traceable]:
    """
    To wrap a module as a basic unit, is to make it a primitive and stop the engine from digging deeper into it.

    ``basic_unit_tag`` is true by default. If set to false, it will not be explicitly mark as a basic unit, and
    graph parser will continue to parse. Currently, this is to handle a special case in ``nn.Sequential``.

    .. code-block:: python

        @basic_unit
        class PrimitiveOp(nn.Module):
            ...
    """
    import torch.nn as nn
    assert issubclass(cls, nn.Module), 'When using @basic_unit, the class must be a subclass of nn.Module.'

    cls = trace(cls)
    cls._nni_basic_unit = basic_unit_tag
    return cls


def model_wrapper(cls: T) -> Union[T, Traceable]:
    """
    Wrap the model if you are using pure-python execution engine. For example

    .. code-block:: python

        @model_wrapper
        class MyModel(nn.Module):
            ...

    The wrapper serves two purposes:

        1. Capture the init parameters of python class so that it can be re-instantiated in another process.
        2. Reset uid in `mutation` namespace so that each model counts from zero.
           Can be useful in unittest and other multi-model scenarios.
    """
    import torch.nn as nn
    assert issubclass(cls, nn.Module)

    wrapper = trace(cls)

    class reset_wrapper(wrapper):
        def __init__(self, *args, **kwargs):
            with ModelNamespace():
                super().__init__(*args, **kwargs)

    reset_wrapper.__dict__['_nni_model_wrapper'] = True
    _copy_class_wrapper_attributes(wrapper, reset_wrapper)
    return reset_wrapper
