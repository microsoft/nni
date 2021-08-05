# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import functools
import inspect
import types
from typing import Any

import json_tricks

from .utils import get_importable_name, get_module_name, import_, reset_uid


def get_init_parameters_or_fail(obj, silently=False):
    if hasattr(obj, '_init_parameters'):
        return obj._init_parameters
    elif silently:
        return None
    else:
        raise ValueError(f'Object {obj} needs to be serializable but `_init_parameters` is not available. '
                         'If it is a built-in module (like Conv2d), please import it from retiarii.nn. '
                         'If it is a customized module, please to decorate it with @basic_unit. '
                         'For other complex objects (e.g., trainer, optimizer, dataset, dataloader), '
                         'try to use serialize or @serialize_cls.')


### This is a patch of json-tricks to make it more useful to us ###


def _serialize_class_instance_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    try:  # FIXME: raise error
        if hasattr(obj, '__class__'):
            return {
                '__type__': get_importable_name(obj.__class__),
                'arguments': get_init_parameters_or_fail(obj)
            }
    except ValueError:
        pass
    return obj


def _serialize_class_instance_decode(obj):
    if isinstance(obj, dict) and '__type__' in obj and 'arguments' in obj:
        return import_(obj['__type__'])(**obj['arguments'])
    return obj


def _type_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    if isinstance(obj, type):
        return {'__typename__': get_importable_name(obj, relocate_module=True)}
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
        # This is not reliable for cases like closure, `open`, or objects that is callable but not intended to be serialized.
        # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
        return {'__typename__': get_importable_name(obj, relocate_module=True)}
    return obj


def _type_decode(obj):
    if isinstance(obj, dict) and '__typename__' in obj:
        return import_(obj['__typename__'])
    return obj


json_loads = functools.partial(json_tricks.loads, extra_obj_pairs_hooks=[_serialize_class_instance_decode, _type_decode])
json_dumps = functools.partial(json_tricks.dumps, extra_obj_encoders=[_serialize_class_instance_encode, _type_encode])
json_load = functools.partial(json_tricks.load, extra_obj_pairs_hooks=[_serialize_class_instance_decode, _type_decode])
json_dump = functools.partial(json_tricks.dump, extra_obj_encoders=[_serialize_class_instance_encode, _type_encode])

### End of json-tricks patch ###


class Translatable(abc.ABC):
    """
    Inherit this class and implement ``translate`` when the inner class needs a different
    parameter from the wrapper class in its init function.
    """

    @abc.abstractmethod
    def _translate(self) -> Any:
        pass


def _create_wrapper_cls(cls, store_init_parameters=True, reset_mutation_uid=False):
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            if reset_mutation_uid:
                reset_uid('mutation')
            if store_init_parameters:
                argname_list = list(inspect.signature(cls.__init__).parameters.keys())[1:]
                full_args = {}
                full_args.update(kwargs)

                assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
                for argname, value in zip(argname_list, args):
                    full_args[argname] = value

                # translate parameters
                args = list(args)
                for i, value in enumerate(args):
                    if isinstance(value, Translatable):
                        args[i] = value._translate()
                for i, value in kwargs.items():
                    if isinstance(value, Translatable):
                        kwargs[i] = value._translate()

                self._init_parameters = full_args
            else:
                self._init_parameters = {}

            super().__init__(*args, **kwargs)

    wrapper.__module__ = get_module_name(cls)
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__init__.__doc__ = cls.__init__.__doc__

    return wrapper


def serialize_cls(cls):
    """
    To create an serializable class.
    """
    return _create_wrapper_cls(cls)


def transparent_serialize(cls):
    """
    Wrap a module but does not record parameters. For internal use only.
    """
    return _create_wrapper_cls(cls, store_init_parameters=False)


def serialize(cls, *args, **kwargs):
    """
    To create an serializable instance inline without decorator. For example,

    .. code-block:: python

        self.op = serialize(MyCustomOp, hidden_units=128)
    """
    return serialize_cls(cls)(*args, **kwargs)


def basic_unit(cls):
    """
    To wrap a module as a basic unit, to stop it from parsing and make it mutate-able.
    """
    import torch.nn as nn
    assert issubclass(cls, nn.Module), 'When using @basic_unit, the class must be a subclass of nn.Module.'
    return serialize_cls(cls)


def model_wrapper(cls):
    """
    Wrap the model if you are using pure-python execution engine.

    The wrapper serves two purposes:

        1. Capture the init parameters of python class so that it can be re-instantiated in another process.
        2. Reset uid in `mutation` namespace so that each model counts from zero. Can be useful in unittest and other multi-model scenarios.
    """
    return _create_wrapper_cls(cls, reset_mutation_uid=True)
