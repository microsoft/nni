import inspect
from collections import defaultdict
from typing import Any


def import_(target: str, allow_none: bool = False) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


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


def _register_module(original_class):
    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    def __init__(self, *args, **kws):
        full_args = {}
        full_args.update(kws)
        for i, arg in enumerate(args):
            full_args[argname_list[i]] = arg
        add_record(id(self), full_args)

        orig_init(self, *args, **kws)  # Call the original __init__

    original_class.__init__ = __init__  # Set the class' __init__ to the new one
    return original_class


def register_module():
    """
    Register a module.
    """
    # use it as a decorator: @register_module()
    def _register(cls):
        m = _register_module(
            original_class=cls)
        return m

    return _register


def _register_trainer(original_class):
    orig_init = original_class.__init__
    argname_list = list(inspect.signature(original_class).parameters.keys())
    # Make copy of original __init__, so we can call it without recursion

    full_class_name = original_class.__module__ + '.' + original_class.__name__

    def __init__(self, *args, **kws):
        full_args = {}
        full_args.update(kws)
        for i, arg in enumerate(args):
            # TODO: support both pytorch and tensorflow
            from .nn.pytorch import Module
            if isinstance(args[i], Module):
                # ignore the base model object
                continue
            full_args[argname_list[i]] = arg
        add_record(id(self), {'modulename': full_class_name, 'args': full_args})

        orig_init(self, *args, **kws)  # Call the original __init__

    original_class.__init__ = __init__  # Set the class' __init__ to the new one
    return original_class


def register_trainer():
    def _register(cls):
        m = _register_trainer(
            original_class=cls)
        return m

    return _register


_last_uid = defaultdict(int)


def uid(namespace: str = 'default') -> int:
    _last_uid[namespace] += 1
    return _last_uid[namespace]
