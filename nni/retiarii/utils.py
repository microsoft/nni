import functools
import inspect
import warnings
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


def _blackbox_class_instance_encode(obj, primitives=False):
    assert not primitives, 'Encoding with primitives is not supported.'
    if hasattr(obj, '__class__') and hasattr(obj, '__init_parameters__'):
        return {
            '__instance_type__': _get_full_class_name(obj.__class__.__module__, obj.__class__.__name__),
            'arguments': obj.__init_parameters__
        }
    return obj


def _blackbox_class_instance_decode(obj):
    if '__instance_type__' in obj and 'arguments' in obj:
        return import_(obj['__instance_type__'])(**obj['arguments'])
    return obj


json_loads = functools.partial(json_tricks.loads, extra_obj_pairs_hooks=[_blackbox_class_instance_decode])
json_dumps = functools.partial(json_tricks.dumps, extra_obj_encoders=[_blackbox_class_instance_encode])
json_load = functools.partial(json_tricks.load, extra_obj_pairs_hooks=[_blackbox_class_instance_decode])
json_dump = functools.partial(json_tricks.dump, extra_obj_encoders=[_blackbox_class_instance_encode])


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


def _blackbox_cls(cls, module_name, register_format=None):
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            argname_list = list(inspect.signature(cls.__init__).parameters.keys())[1:]
            full_args = {}
            full_args.update(kwargs)

            assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
            for argname, value in zip(argname_list, args):
                full_args[argname] = value

            if register_format == 'args':
                add_record(id(self), full_args)
            elif register_format == 'full':
                full_class_name = _get_full_class_name(cls.__module__, cls.__name__)
                add_record(id(self), {'modulename': full_class_name, 'args': full_args})

            self.__init_parameters__ = full_args

            super().__init__(*args, **kwargs)

        def __del__(self):
            del_record(id(self))

    # using module_name instead of cls.__module__ because it's more natural to see where the module gets wrapped
    # instead of simply putting torch.nn or etc.
    wrapper.__module__ = module_name
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
    return _blackbox_cls(cls, _get_module_name(cls, inspect.stack(), 'blackbox'), 'args')(*args, **kwargs)


def blackbox_module(cls):
    """
    Register a module. Use it as a decorator.
    """
    return _blackbox_cls(cls, _get_module_name(cls, inspect.stack(), 'module'), 'args')


def register_trainer(cls):
    """
    Register a trainer. Use it as a decorator.
    """
    return _blackbox_cls(cls, _get_module_name(cls, inspect.stack(), 'trainer'), 'full')


_last_uid = defaultdict(int)


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]


def _get_module_name(cls, inspect_stack, placeholder):
    frm = inspect_stack[1]
    module_name = cls.__module__
    if module_name == '__main__':
        main_file_path = Path(inspect.getsourcefile(frm[0]))
        if main_file_path.parents[0] != Path('.'):
            raise RuntimeError(f'You are using "{main_file_path}" to launch your experiment, '
                               f'please launch the experiment under the directory where "{main_file_path.name}" is located.')
        module_name = main_file_path.stem
    return module_name


def _get_full_class_name(module_name, class_name):
    return module_name + '.' + class_name
