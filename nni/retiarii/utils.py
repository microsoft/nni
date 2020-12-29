import inspect
from collections import defaultdict
from typing import Any


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


_records = {}


def get_records():
    global _records
    return _records


def add_record(key, value):
    """
    """
    global _records
    if _records is not None:
        assert key not in _records, '{} already in _records'.format(key)
        _records[key] = value


def _blackbox_cls(cls, register_format=None):
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            argname_list = list(inspect.signature(cls).parameters.keys())
            full_args = {}
            full_args.update(kwargs)

            assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
            for argname, value in zip(argname_list, args):
                full_args[argname] = value

            # eject un-serializable arguments
            for k in full_args:
                if not isinstance(full_args[k], (int, float, str, dict, list)):
                    full_args.pop(k)

            if register_format == 'args':
                add_record(id(self), full_args)
            elif register_format == 'full':
                full_class_name = cls.__module__ + '.' + cls.__name__
                add_record(id(self), {'modulename': full_class_name, 'args': full_args})

            super().__init__(*args, **kwargs)

        def __del__(self):
            raise RuntimeError(f'Blackbox class instance {str(self)} should not be deleted.')

    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__qualname__
    wrapper.__init__.__doc__ = cls.__init__.__doc__

    return wrapper


def blackbox(cls, *args, **kwargs):
    return _blackbox_cls(cls, 'args')(*args, **kwargs)


def blackbox_module(cls):
    """
    Register a module. Use it as a decorator.
    """
    return _blackbox_cls(cls, 'args')


def blackbox_trainer(cls):
    """
    Register a trainer. Use it as a decorator.
    """
    return _blackbox_cls(cls, 'full')


_last_uid = defaultdict(int)


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]
