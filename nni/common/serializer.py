import base64
import functools
import inspect
from typing import Any, Union, Dict, Optional, List, TypeVar

import json_tricks  # use json_tricks as serializer backend
import cloudpickle  # use cloudpickle as backend for unserializable types and instances


__all__ = ['trace', 'dump', 'load', 'SerializableObject']


T = TypeVar('T')


class SerializableObject:
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.
    """

    def __init__(self, symbol: T, args: List[Any], kwargs: Dict[str, Any],
                 _self_contained: bool = False):
        # use dict to avoid conflicts with user's getattr and setattr
        self.__dict__['_nni_symbol'] = symbol
        self.__dict__['_nni_args'] = args
        self.__dict__['_nni_kwargs'] = kwargs

        self.__dict__['_nni_self_contained'] = _self_contained

        if _self_contained:
            # this is for internal usage only.
            # kwargs is used to init the full object in the same object as this one, for simpler implementation.
            super().__init__(*self._recursive_init(args), **self._recursive_init(kwargs))

    def get(self) -> Any:
        """
        Get the original object.
        """
        if self._get_nni_attr('self_contained'):
            return self
        if '_nni_cache' not in self.__dict__:
            self.__dict__['_nni_cache'] = self._get_nni_attr('symbol')(
                *self._recursive_init(self._get_nni_attr('args')),
                **self._recursive_init(self._get_nni_attr('kwargs'))
            )
        return self.__dict__['_nni_cache']

    def copy(self) -> Union[T, 'SerializableObject']:
        """
        Perform a shallow copy. Will throw away the self-contain property for classes (refer to implementation).
        This is the one that should be used when you want to "mutate" a serializable object.
        """
        return SerializableObject(
            self._get_nni_attr('symbol'),
            self._get_nni_attr('args'),
            self._get_nni_attr('kwargs')
        )

    def __json_encode__(self):
        ret = {'__symbol__': _get_hybrid_cls_or_func_name(self._get_nni_attr('symbol'))}
        if self._get_nni_attr('args'):
            ret['__args__'] = self._get_nni_attr('args')
        ret['__kwargs__'] = self._get_nni_attr('kwargs')
        return ret

    def _get_nni_attr(self, name):
        return self.__dict__['_nni_' + name]

    def __repr__(self):
        if self._get_nni_attr('self_contained'):
            return repr(self)
        if '_nni_cache' in self.__dict__:
            return repr(self._get_nni_attr('cache'))
        return 'SerializableObject(' + \
            ', '.join(['type=' + self._get_nni_attr('symbol').__name__] +
                      [repr(d) for d in self._get_nni_attr('args')] +
                      [k + '=' + repr(v) for k, v in self._get_nni_attr('kwargs').items()]) + \
            ')'

    @staticmethod
    def _recursive_init(d):
        # auto-call get() to prevent type-converting in downstreaming functions
        if isinstance(d, dict):
            return {k: v.get() if isinstance(v, SerializableObject) else v for k, v in d.items()}
        else:
            return [v.get() if isinstance(v, SerializableObject) else v for v in d]


def trace(cls_or_func: T = None, *, kw_only: bool = True) -> Union[T, SerializableObject]:
    """
    Annotate a function or a class if you want to preserve where it comes from.
    This is usually used in the following scenarios:

    1) Care more about execution configuration rather than results, which is usually the case in AutoML. For example,
       you want to mutate the parameters of a function.
    2) Repeat execution is not an issue (e.g., reproducible, execution is fast without side effects).

    When a class/function is annotated, all the instances/calls will return a object as it normally will.
    Although the object might act like a normal object, it's actually a different object with NNI-specific properties.
    To get the original object, you should use ``obj.get()`` to retrieve. The retrieved object can be used
    like the original one, but there are still subtle differences in implementation.

    Note that when using the result from a trace in another trace-able function/class, ``.get()`` is automatically
    called, so that you don't have to worry about type-converting.

    Also it records extra information about where this object comes from. That's why it's called "trace".
    When call ``nni.dump``, that information will be used, by default.

    If ``kw_only`` is true, try to convert all parameters into kwargs type. This is done by inspect the argument
    list and types. This can be useful to extract semantics, but can be tricky in some corner cases.

    Example:

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass
    """

    def wrap(cls_or_func):
        if isinstance(cls_or_func, type):
            return _trace_cls(cls_or_func, kw_only)
        else:
            return _trace_func(cls_or_func, kw_only)

    # if we're being called as @trace()
    if cls_or_func is None:
        return wrap

    # if we are called without parentheses
    return wrap(cls_or_func)


def dump(obj: Any, fp: Optional[Any] = None, use_trace: bool = True, pickle_size_limit: int = 4096,
         **json_tricks_kwargs) -> Union[str, bytes]:
    """
    Convert a nested data structure to a json string. Save to file if fp is specified.
    Use json-tricks as main backend. For unhandled cases in json-tricks, use cloudpickle.
    The serializer is not designed for long-term storage use, but rather to copy data between processes.
    The format is also subject to change between NNI releases.

    Parameters
    ----------
    fp : file handler or path
        File to write to. Keep it none if you want to dump a string.
    pickle_size_limit : int
        This is set to avoid too long serialization result. Set to -1 to disable size check.
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
        functools.partial(_json_tricks_serializable_object_encode, use_trace=use_trace),
        functools.partial(_json_tricks_func_or_cls_encode, pickle_size_limit=pickle_size_limit),
        functools.partial(_json_tricks_any_object_encode, pickle_size_limit=pickle_size_limit),
    ]

    if fp is not None:
        return json_tricks.dump(obj, fp, obj_encoders=encoders, **json_tricks_kwargs)
    else:
        return json_tricks.dumps(obj, obj_encoders=encoders, **json_tricks_kwargs)


def load(string: str = None, fp: Optional[Any] = None, **json_tricks_kwargs) -> Any:
    """
    Load the string or from file, and convert it to a complex data structure.
    At least one of string or fp has to be not none.

    Parameters
    ----------
    """
    assert string is not None or fp is not None
    # see encoders for explanation
    hooks = [
        json_tricks.pathlib_hook,
        json_tricks.pandas_hook,
        json_tricks.json_numpy_obj_hook,
        json_tricks.decoders.EnumInstanceHook(),
        json_tricks.json_date_time_hook,
        json_tricks.json_complex_hook,
        json_tricks.json_set_hook,
        json_tricks.numeric_types_hook,
        _json_tricks_serializable_object_decode,
        _json_tricks_func_or_cls_decode,
        _json_tricks_any_object_decode
    ]

    if string is not None:
        return json_tricks.loads(string, obj_pairs_hooks=hooks, **json_tricks_kwargs)
    else:
        return json_tricks.load(fp, obj_pairs_hooks=hooks, **json_tricks_kwargs)


def _trace_cls(base, kw_only):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    class wrapper(SerializableObject, base):
        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            args, kwargs = _get_arguments_as_dict(base.__init__, args, kwargs, kw_only)

            # calling serializable object init to initialize the full object
            super().__init__(symbol=base, args=args, kwargs=kwargs, _self_contained=True)

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


def _trace_func(func, kw_only):
    @functools.wraps
    def wrapper(*args, **kwargs):
        # similar to class, store parameters here
        args, kwargs = _get_arguments_as_dict(func, args, kwargs, kw_only)
        return SerializableObject(func, args, kwargs)

    return wrapper


def _get_arguments_as_dict(func, args, kwargs, kw_only):
    if kw_only:
        # get arguments passed to a function, and save it as a dict
        argname_list = list(inspect.signature(func).parameters.keys())[1:]
        full_args = {}
        full_args.update(kwargs)

        # match arguments with given arguments
        # args should be longer than given list, because args can be used in a kwargs way
        assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
        for argname, value in zip(argname_list, args):
            full_args[argname] = value

        args, kwargs = [], full_args

    return args, kwargs


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

    return full_name


def _get_hybrid_cls_or_func_name(cls_or_func: Any, pickle_size_limit: int = 4096) -> str:
    try:
        name = _get_cls_or_func_name(cls_or_func)
        # import success, use a path format
        return 'path:' + name
    except ImportError:
        b = cloudpickle.dumps(cls_or_func)
        if len(b) > pickle_size_limit:
            raise ValueError(f'Pickle too large when trying to dump {cls_or_func}. '
                             'Please try to raise pickle_size_limit if you insist.')
        # fallback to cloudpickle
        return 'bytes:' + base64.b64encode(b).decode()


def _import_cls_or_func_from_hybrid_name(s: str) -> Any:
    if s.startswith('bytes:'):
        b = base64.b64decode(s.split(':', 1)[-1])
        return cloudpickle.loads(b)
    if s.startswith('path:'):
        s = s.split(':', 1)[-1]
    return _import_cls_or_func_from_name(s)


def _json_tricks_func_or_cls_encode(cls_or_func: Any, primitives: bool = False, pickle_size_limit: int = 4096) -> str:
    if not isinstance(cls_or_func, type) and not callable(cls_or_func):
        # not a function or class, continue
        return cls_or_func

    return {
        '__nni_type__': _get_hybrid_cls_or_func_name(cls_or_func, pickle_size_limit)
    }


def _json_tricks_func_or_cls_decode(s: Dict[str, Any]) -> Any:
    if isinstance(s, dict) and '__nni_type__' in s:
        s = s['__nni_type__']
        return _import_cls_or_func_from_hybrid_name(s)
    return s


def _json_tricks_serializable_object_encode(obj: Any, primitives: bool = False, use_trace: bool = True) -> Dict[str, Any]:
    # Encodes a serializable object instance to json.
    # If primitives, the representation is simplified and cannot be recovered!

    # do nothing to instance that is not a serializable object and do not use trace
    if not use_trace or not isinstance(obj, SerializableObject):
        return obj

    return obj.__json_encode__()


def _json_tricks_serializable_object_decode(obj: Dict[str, Any]) -> Any:
    if isinstance(obj, dict) and '__symbol__' in obj and '__kwargs__' in obj:
        return SerializableObject(
            _import_cls_or_func_from_hybrid_name(obj['__symbol__']),
            getattr(obj, '__args__', []),
            obj['__kwargs__']
        )
    return obj


def _json_tricks_any_object_encode(obj: Any, primitives: bool = False, pickle_size_limit: int = 4096) -> Any:
    # We want to use this to replace the class instance encode in json-tricks.
    # Therefore the coverage should be roughly same.
    if isinstance(obj, list) or isinstance(obj, dict):
        return obj
    if hasattr(obj, '__class__') and (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')):
        b = cloudpickle.dumps(obj)
        if len(b) > pickle_size_limit:
            raise ValueError(f'Pickle too large when trying to dump {obj}. '
                             'Please try to raise pickle_size_limit if you insist.')
        # use base64 to dump a bytes array
        return {
            '__nni_obj__': base64.b64encode(b).decode()
        }
    return obj


def _json_tricks_any_object_decode(obj: Dict[str, Any]) -> Any:
    if isinstance(obj, dict) and '__nni_obj__' in obj:
        obj = obj['__nni_obj__']
        b = base64.b64decode(obj)
        return cloudpickle.loads(b)
    return obj
