import functools
import inspect
from typing import Any, Callable, Union, Type, Dict

from .utils import get_importable_name, get_module_name, import_, reset_uid


__all__ = ['trace', 'dump', 'load', 'SerializableObject']


def trace(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
    """
    Annotate a function or a class if you want to preserve where it comes from.
    This is usually used in the following scenarios:

    1) Care more about execution configuration rather than results, which is usually the case in AutoML.
    2) Repeat execution is not an issue (e.g., reproducible, execution is fast without side effects).

    When a class/function is annotated, all the instances/calls will return a object as it normally will.
    Although the object might act like a normal object, it's actually a different object with NNI-specific properties.
    To get the original object, you should use ``obj.get()`` to retrieve. The retrieved object can be used
    like the original one, but there are still subtle differences in implementation.

    Also it records extra information about where this object comes from. That's why it's called "trace".
    When call ``nni.dump``, that information will be used, by default.
    """

    if isinstance(cls_or_func, type):
        return _trace_cls(cls_or_func)
    else:
        return _trace_func(cls_or_func)


def dump(obj, f=None, use_trace=True): ...


def load(string=None, f=None): ...


class SerializableObject:
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.
    """

    def __init__(self, nni_symbol: Union[Type, Callable], nni_args: Dict[str, Any], _internal: bool = False, **kwargs):
        self._nni_symbol = nni_symbol
        self._nni_args = nni_args

        # kwargs is used to init the full object.
        if not _internal:
            assert not kwargs, 'kwargs cannot be set for non-internal usage.'
        super().__init__(**kwargs)

    def get(self) -> Any:
        """
        Get the original object.
        """
        return self  # for class cases

    def dump(self):
        return {
            'symbol': self._nni_symbol,
            'args': self._nni_args
        }


class SerializableFunctionCall(SerializableObject):
    """
    Serializable function call. Can be created via:

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass

    Then call ``foo(1)`` will return a object with this type and need to call foo(1).get() to get the actual result.
    """

    def get(self):
        if not hasattr(self, '_nni_func_cache'):
            self._nni_func_cache = self.symbol(self.args)()
        return self._nni_func_cache


def _trace_cls(base):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    class wrapper(SerializableObject, base):

        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            full_args = _get_arguments_as_dict(base.__init__, args, kwargs)

            # calling serializable object init to initialize the full object
            super().__init__(nni_symbol=base, nni_args=full_args, _internal=True, **full_args)

    _MISSING = '_missing'
    for k in functools.WRAPPER_ASSIGNMENTS:
        # assign magic attributes like __module__, __qualname__, __doc__
        v = getattr(base, k, _MISSING)
        if v is not _MISSING:
            try:
                setattr(wrapper, k, v)
            except AttributeError:
                pass

    return wrapper


def _trace_func(func):
    @functools.wraps
    def wrapper(*args, **kwargs):
        # similar to class, store parameters here
        full_args = _get_arguments_as_dict(func, args, kwargs)
        return SerializableFunctionCall(func, full_args)

    return wrapper


def _get_arguments_as_dict(func, args, kwargs):
    # get arguments passed to a function, and save it as a dict
    argname_list = list(inspect.signature(func).parameters.keys())[1:]
    full_args = {}
    full_args.update(kwargs)

    # match arguments with given arguments
    # args should be longer than given list, because args can be used in a kwargs way
    assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
    for argname, value in zip(argname_list, args):
        full_args[argname] = value

    return full_args
