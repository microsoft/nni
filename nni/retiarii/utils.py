import inspect
import warnings
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


def clear_records():
    global _records
    _records = {}


def add_record(key, value):
    """
    """
    global _records
    if _records is not None:
        # assert key not in _records, '{} already in _records'.format(key)  # FIXME temporarily comment this line
        _records[key] = value


def del_record(key):
    global _records
    if _records is not None:
        _records.pop(key, None)


def _blackbox_cls(cls, module_name, register_format=None):
    class wrapper(cls):
        def __init__(self, *args, **kwargs):
            argname_list = list(inspect.signature(cls).parameters.keys())
            full_args = {}
            full_args.update(kwargs)

            assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
            for argname, value in zip(argname_list, args):
                full_args[argname] = value

            # eject un-serializable arguments
            for k in list(full_args.keys()):
                # The list is not complete and does not support nested cases.
                if not isinstance(full_args[k], (int, float, str, dict, list)):
                    if not (register_format == 'full' and k == 'model'):
                        # no warning if it is base model in trainer
                        warnings.warn(f'{cls} has un-serializable arguments {k} whose value is {full_args[k]}. \
                            This is not supported. You can ignore this warning if you are passing the model to trainer.')
                    full_args.pop(k)

            if register_format == 'args':
                add_record(id(self), full_args)
            elif register_format == 'full':
                full_class_name = cls.__module__ + '.' + cls.__name__
                add_record(id(self), {'modulename': full_class_name, 'args': full_args})

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
    # get caller module name
    frm = inspect.stack()[1]
    module_name = inspect.getmodule(frm[0]).__name__
    return _blackbox_cls(cls, module_name, 'args')(*args, **kwargs)


def blackbox_module(cls):
    """
    Register a module. Use it as a decorator.
    """
    frm = inspect.stack()[1]
    module_name = inspect.getmodule(frm[0]).__name__
    return _blackbox_cls(cls, module_name, 'args')


def register_trainer(cls):
    """
    Register a trainer. Use it as a decorator.
    """
    frm = inspect.stack()[1]
    module_name = inspect.getmodule(frm[0]).__name__
    return _blackbox_cls(cls, module_name, 'full')


_last_uid = defaultdict(int)


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]
