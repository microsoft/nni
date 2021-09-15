import base64
import functools
import inspect
from typing import Any, Callable, Union, Type, Dict, Optional

import json_tricks  # use json_tricks as serializer backend
import cloudpickle  # use cloudpickle as backend for unserializable types and instances


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

    Note that when using the result from a trace in another trace-able function/class, ``.get()`` is automatically
    called, so that you don't have to worry about type-converting.

    Also it records extra information about where this object comes from. That's why it's called "trace".
    When call ``nni.dump``, that information will be used, by default.

    Example:

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass
    """

    if isinstance(cls_or_func, type):
        return _trace_cls(cls_or_func)
    else:
        return _trace_func(cls_or_func)


def dump(obj: Any, fp: Optional[Any] = None, use_trace: bool = True, **json_tricks_kwargs) -> Union[str, bytes]:
    """
    Parameters
    ----------
    fp : file handler or path
        File to write to. Keep it none if you want to dump a string.
    json_tricks_kwargs : dict
        Other keyword arguments passed to json tricks (backend), e.g., indent=2.

    Returns
    -------
    str or bytes
        Normally str. Sometimes bytes (if compressed).
    """

    encoders = [
        # we don't need to check for dependency as many of those have already been required by NNI
        json_tricks.pathlib_encode,         # pathlib is a required dependency for NNI
        json_tricks.pandas_encode,          # pandas is a required dependency
        json_tricks.numpy_encode,           # required
        json_tricks.encoders.enum_instance_encode,
        json_tricks.json_date_time_encode,  # same as json_tricks
        json_tricks.json_complex_encode,
        json_tricks.json_set_encode,
        json_tricks.numeric_types_encode,
        functools.partial(serializable_object_encode, use_trace=use_trace),
    ]

    if fp is not None:
        return json_tricks.dump(obj, fp, obj_encoders=encoders, **json_tricks_kwargs)
    else:
        return json_tricks.dumps(obj, **json_tricks_kwargs)


def load(string=None, f=None):
    # see encoders for explanation
    hook = [
        json_tricks.pathlib_hook,
        json_tricks.pandas_hook,
        json_tricks.json_numpy_obj_hook,
        json_tricks.decoders.EnumInstanceHook(),
        json_tricks.json_date_time_hook,
        json_tricks.json_complex_hook,
        json_tricks.json_set_hook,
        json_tricks.numeric_types_hook,
        class_hook,
    ]


class SerializableObject:
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.
    """

    def __init__(self, nni_symbol: Union[Type, Callable], nni_args: Dict[str, Any], _self_contained: bool = False, **kwargs):
        self._nni_symbol = nni_symbol
        self._nni_args = nni_args

        self._self_contained = _self_contained

        if not _self_contained:
            assert not kwargs, 'kwargs cannot be set for non-internal usage.'
        else:
            # this is for internal usage only.
            # kwargs is used to init the full object in the same object as this one, for simpler implementation.
            super().__init__(**kwargs)

    def get(self) -> Any:
        """
        Get the original object.
        """
        if self._self_contained:
            return self
        if not hasattr(self, '_nni_cache'):
            self._nni_cache = self.symbol(self.args)()
        return self._nni_cache

    def copy(self) -> 'SerializableObject':
        """
        Perform a shallow copy. Will throw away the self-contain property for classes (refer to implementation).
        This is the one that should be used when you want to "mutate" a serializable object.
        """
        return SerializableObject(self._nni_symbol, self._nni_args)

    def dump(self):
        return {
            '__symbol__': self._nni_symbol,
            '__args__': self._nni_args
        }


def _trace_cls(base):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    class wrapper(SerializableObject, base):

        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            full_args = _get_arguments_as_dict(base.__init__, args, kwargs)

            # calling serializable object init to initialize the full object
            super().__init__(nni_symbol=base, nni_args=full_args, _self_contained=True, **full_args)

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
        return SerializableObject(func, full_args)

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

    # auto-call get() to prevent type-converting in downstreaming functions
    kwargs = {k: v.get() if isinstance(v, SerializableObject) else v for k, v in kwargs.items()}

    return full_args


def _import_cls_or_func_from_name(target: str) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


def _get_cls_or_func_name(cls_or_func: Any) -> str:
    module_name = cls_or_func.__module__
    if module_name == '__main__':
        raise ImportError('Cannot use a path to identify something from __main__.')
    full_name = module_name + '.' + cls_or_func.__name__

    try:
        imported = _import_cls_or_func_from_name(full_name)
        if imported != cls_or_func:
            raise ImportError(f'Imported {imported} is not same as expected. The function might be dynamically created.')
    except ImportError:
        raise ImportError(f'Import {cls_or_func.__name__} from "{module_name}" failed.')

    return module_name


def _json_tricks_func_or_cls_encode(cls_or_func: Any, primitives: bool = False) -> str:
    try:
        name = _get_cls_or_func_name(cls_or_func)
        return 'nni-type:path:' + name
    except ImportError:
        b = cloudpickle.dumps(cls_or_func)
        return 'nni-type:bytes:' + base64.b64encode(b).decode()


def _json_tricks_func_or_cls_decode(s: str) -> Any:
    if isinstance(s, str):
        if s.startswith('nni-type:path:'):
            return _import_cls_or_func_from_name(s.split(':', 2)[-1])
        elif s.startswith('nni-type:bytes:'):
            b = base64.b64decode(s.split(':', 2)[-1])
            return cloudpickle.loads(b)
    return s


def _json_tricks_serializable_object_encode(obj, primitives=False, use_trace=True):
    # Encodes a serializable object instance to json.
    # If primitives, the representation is simplified and cannot be recovered!

    # do nothing to instance that is not a serializable object
    if not isinstance(obj, SerializableObject):
        return obj


    if hasattr(obj, '__class__') and (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')):
        if not hasattr(obj, '__new__'):
            raise TypeError('class "{0:s}" does not have a __new__ method; '.format(obj.__class__) +
                ('perhaps it is an old-style class not derived from `object`; add `object` as a base class to encode it.'
                    if (version[:2] == '2.') else 'this should not happen in Python3'))
        if type(obj) == type(lambda: 0):
            raise TypeError('instance "{0:}" of class "{1:}" cannot be encoded because it appears to be a lambda or function.'
                .format(obj, obj.__class__))
        try:
            obj.__new__(obj.__class__)
        except TypeError:
            raise TypeError(('instance "{0:}" of class "{1:}" cannot be encoded, perhaps because it\'s __new__ method '
                'cannot be called because it requires extra parameters').format(obj, obj.__class__))
        mod = get_module_name_from_object(obj)
        if mod == 'threading':
            # In Python2, threading objects get serialized, which is probably unsafe
            return obj
        name = obj.__class__.__name__
        if hasattr(obj, '__json_encode__'):
            attrs = obj.__json_encode__()
            if primitives:
                return attrs
            else:
                return hashodict((('__instance_type__', (mod, name)), ('attributes', attrs)))
        dct = hashodict([('__instance_type__',(mod, name))])
        if hasattr(obj, '__slots__'):
            slots = obj.__slots__
            if isinstance(slots, str):
                slots = [slots]
            slots = list(item for item in slots if item != '__dict__')
            dct['slots'] = hashodict([])
            for s in slots:
                dct['slots'][s] = getattr(obj, s)
        if hasattr(obj, '__dict__'):
            dct['attributes'] = hashodict(obj.__dict__)
        if primitives:
            attrs = dct.get('attributes',{})
            attrs.update(dct.get('slots',{}))
            return attrs
        else:
            return dct
    return obj
