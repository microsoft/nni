import abc
import functools
import inspect
from typing import Any

import json_tricks

from .utils import get_full_class_name, get_module_name, import_

### This is a patch of json-tricks to make it more useful to us ###


def _serialize_class_instance_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    if hasattr(obj, '__class__') and hasattr(obj, '__init_parameters__'):
        return {
            '__type__': get_full_class_name(obj.__class__),
            'arguments': obj.__init_parameters__
        }
    return obj


def _serialize_class_instance_decode(obj):
    if isinstance(obj, dict) and '__type__' in obj and 'arguments' in obj:
        return import_(obj['__type__'])(**obj['arguments'])
    return obj


def _type_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    if isinstance(obj, type):
        return {'__typename__': get_full_class_name(obj, relocate_module=True)}
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


def serialize_cls(cls):
    """
    To create an serializable class.
    """
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
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

            self.__init_parameters__ = full_args

            super().__init__(*args, **kwargs)

    wrapper.__module__ = get_module_name(cls)
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__init__.__doc__ = cls.__init__.__doc__

    return wrapper


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
