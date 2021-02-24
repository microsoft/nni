import abc
import functools
import inspect
from collections import defaultdict
from typing import Any
from pathlib import Path

import json_tricks


def import_(target: str, allow_none: bool = False) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


def version_larger_equal(a: str, b: str) -> bool:
    # TODO: refactor later
    a = a.split('+')[0]
    b = b.split('+')[0]
    return tuple(map(int, a.split('.'))) >= tuple(map(int, b.split('.')))


### This is a patch of json-tricks to make it more useful to us ###

def _blackbox_class_instance_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    if hasattr(obj, '__class__') and hasattr(obj, '__init_parameters__'):
        return {
            '__type__': get_full_class_name(obj.__class__),
            'arguments': obj.__init_parameters__
        }
    return obj


def _blackbox_class_instance_decode(obj):
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


json_loads = functools.partial(json_tricks.loads, extra_obj_pairs_hooks=[_blackbox_class_instance_decode, _type_decode])
json_dumps = functools.partial(json_tricks.dumps, extra_obj_encoders=[_blackbox_class_instance_encode, _type_encode])
json_load = functools.partial(json_tricks.load, extra_obj_pairs_hooks=[_blackbox_class_instance_decode, _type_decode])
json_dump = functools.partial(json_tricks.dump, extra_obj_encoders=[_blackbox_class_instance_encode, _type_encode])

### End of json-tricks patch ###


_records = {}


def get_records():
    global _records
    return _records


def clear_records():
    global _records
    _records = {}


def add_record(key, value):
    """
    """
    global _records
    if _records is not None:
        assert key not in _records, f'{key} already in _records. Conflict: {_records[key]}'
        _records[key] = value


def del_record(key):
    global _records
    if _records is not None:
        _records.pop(key, None)


class Translatable(abc.ABC):
    """
    Inherit this class and implement ``translate`` when the inner class needs a different
    parameter from the wrapper class in its init function.
    """

    @abc.abstractmethod
    def _translate(self) -> Any:
        pass


def _blackbox_cls(cls):
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

            add_record(id(self), full_args)  # for compatibility. Will remove soon.

            self.__init_parameters__ = full_args

            super().__init__(*args, **kwargs)

        def __del__(self):
            del_record(id(self))

    wrapper.__module__ = _get_module_name(cls)
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__init__.__doc__ = cls.__init__.__doc__

    return wrapper


def blackbox(cls, *args, **kwargs):
    """
    To create an blackbox instance inline without decorator. For example,

    .. code-block:: python
        self.op = blackbox(MyCustomOp, hidden_units=128)
    """
    return _blackbox_cls(cls)(*args, **kwargs)


def blackbox_module(cls):
    """
    Register a module. Use it as a decorator.
    """
    return _blackbox_cls(cls)


def register_trainer(cls):
    """
    Register a trainer. Use it as a decorator.
    """
    return _blackbox_cls(cls)


_last_uid = defaultdict(int)


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]


def _get_module_name(cls):
    module_name = cls.__module__
    if module_name == '__main__':
        # infer the module name with inspect
        for frm in inspect.stack():
            if inspect.getmodule(frm[0]).__name__ == '__main__':
                # main module found
                main_file_path = Path(inspect.getsourcefile(frm[0]))
                if main_file_path.parents[0] != Path('.'):
                    raise RuntimeError(f'You are using "{main_file_path}" to launch your experiment, '
                                    f'please launch the experiment under the directory where "{main_file_path.name}" is located.')
                module_name = main_file_path.stem
                break
    return module_name


def get_full_class_name(cls, relocate_module=False):
    module_name = _get_module_name(cls) if relocate_module else cls.__module__
    return module_name + '.' + cls.__name__
