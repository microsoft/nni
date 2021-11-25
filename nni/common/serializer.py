import abc
import base64
import functools
import inspect
import numbers
from typing import Any, Union, Dict, Optional, List, TypeVar, Callable
try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import json_tricks  # use json_tricks as serializer backend
import cloudpickle  # use cloudpickle as backend for unserializable types and instances


__all__ = ['trace', 'dump', 'load', 'SerializableObject']


T = TypeVar('T')

class TraceDictType(TypedDict):
    __symbol__: str
    __args__: Optional[List[Any]]
    __kwargs__: Optional[Dict[str, Any]]


class Traceable(abc.ABC):
    """
    A traceable object have copy and dict. Copy and mutate are used to copy the object for further mutations.
    Dict returns a TraceDictType to enable serialization.
    """
    @abc.abstractmethod
    def _trace_copy(self) -> 'Traceable':
        ...

    @abc.abstractmethod
    def _trace_dict(self) -> TraceDictType:
        ...

    @abc.abstractmethod
    def _trace_mutate(self, symbol: str, args: List[Any], kwargs: Dict[str, Any]) -> None:
        ...


class SerializableObject(Traceable):
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.
    """

    def __init__(self, symbol: T, args: List[Any], kwargs: Dict[str, Any],
                 extra_argument_process: Optional[Callable[[Any], Any]] = None):
        # use dict to avoid conflicts with user's getattr and setattr
        self.__dict__['_nni_symbol'] = symbol
        self.__dict__['_nni_args'] = args
        self.__dict__['_nni_kwargs'] = kwargs

        # argument process is another layer to process arguments before they are passed to the underlying class/function.
        # by default, it's simply a `.get()` for serializable object.
        # This is needed because sometimes the recorded arguments are meant to be different from what the inner object receives.
        self.__dict__['_nni_extra_argument_process'] = extra_argument_process

        super().__init__(
            *[_argument_processor(arg, extra_argument_process) for arg in args],
            **{kw: _argument_processor(arg, extra_argument_process) for kw, arg in kwargs.items()}
        )

    def _trace_copy(self) -> Union[T, 'SerializableObject']:
        """
        Perform a shallow copy. Will reinstantiate the class.
        This is the one that should be used when you want to "mutate" a serializable object.
        """
        return SerializableObject(
            self._get_nni_attr('symbol'),
            self._get_nni_attr('args'),
            self._get_nni_attr('kwargs'),
            self._get_nni_attr('extra_argument_process')
        )

    def _trace_dict(self) -> TraceDictType:
        ret = {'__symbol__': _get_hybrid_cls_or_func_name(self._get_nni_attr('symbol'))}
        if self._get_nni_attr('args'):
            ret['__args__'] = self._get_nni_attr('args')
        ret['__kwargs__'] = self._get_nni_attr('kwargs')
        return ret

    def _get_nni_attr(self, name):
        return self.__dict__['_nni_' + name]


    def __repr__(self):
        if self._get_nni_attr('self_contained'):
            return super().__repr__()
        if '_nni_cache' in self.__dict__:
            return repr(self._get_nni_attr('cache'))
        return 'SerializableObject(' + \
            ', '.join(['type=' + self._get_nni_attr('symbol').__name__] +
                      [repr(d) for d in self._get_nni_attr('args')] +
                      [k + '=' + repr(v) for k, v in self._get_nni_attr('kwargs').items()]) + \
            ')'


def trace(cls_or_func: T = None, *, kw_only: bool = True,
          extra_arg_proc: Optional[Callable[[Any], Any]] = None) -> Union[T, SerializableObject]:
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

    If ``kw_only`` is true, try to convert all parameters into kwargs type. This is done by inspecting the argument
    list and types. This can be useful to extract semantics, but can be tricky in some corner cases.

    ``extra_arg_proc`` is used to intercept the arguments for the class or function, and transform them to make them
    different from what they originally received.

    Example:

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass
    """

    def wrap(cls_or_func):
        if isinstance(cls_or_func, type):
            return _trace_cls(cls_or_func, kw_only, extra_arg_proc)
        else:
            return _trace_func(cls_or_func, kw_only, extra_arg_proc)

    # if we're being called as @trace()
    if cls_or_func is None:
        return wrap

    # if we are called without parentheses
    return wrap(cls_or_func)


def dump(obj: Any, fp: Optional[Any] = None, *, use_trace: bool = True, pickle_size_limit: int = 4096,
         allow_nan: bool = True, **json_tricks_kwargs) -> Union[str, bytes]:
    """
    Convert a nested data structure to a json string. Save to file if fp is specified.
    Use json-tricks as main backend. For unhandled cases in json-tricks, use cloudpickle.
    The serializer is not designed for long-term storage use, but rather to copy data between processes.
    The format is also subject to change between NNI releases.

    Parameters
    ----------
    obj : any
        The object to dump.
    fp : file handler or path
        File to write to. Keep it none if you want to dump a string.
    pickle_size_limit : int
        This is set to avoid too long serialization result. Set to -1 to disable size check.
    allow_nan : bool
        Whether to allow nan to be serialized. Different from default value in json-tricks, our default value is true.
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

    json_tricks_kwargs['allow_nan'] = allow_nan

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
    string : str
        JSON string to parse. Can be set to none if fp is used.
    fp : str
        File path to load JSON from. Can be set to none if string is used.

    Returns
    -------
    any
        The loaded object.
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


def _trace_cls(base, kw_only, extra_arg_proc):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    class wrapper(SerializableObject, base):
        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            args, kwargs = _formulate_arguments(base.__init__, args, kwargs, kw_only)

            # calling serializable object init to initialize the full object
            super().__init__(symbol=base, args=args, kwargs=kwargs, _self_contained=True, _extra_argument_process=extra_arg_proc)

    _copy_class_wrapper_attributes(base, wrapper)

    return wrapper


def _trace_func(func, kw_only, extra_arg_proc):
    @functools.wraps
    def wrapper(*args, **kwargs):
        # similar to class, store parameters here
        args, kwargs = _formulate_arguments(func, args, kwargs, kw_only)

        # it's not clear whether this wrapper can handle all the types in python
        # There are many cases here: https://docs.python.org/3/reference/datamodel.html
        # but it looks that we have handled most commonly used cases
        res = func(
            *[_argument_processor(arg) for arg in args],
            **{kw: _argument_processor(arg) for kw, arg in kwargs.items()}
        )

        if res is None:
            res = SerializableObject(func, args, kwargs)
        elif hasattr(res, '__class__') and (hasattr(res, '__dict__') or hasattr(res, '__slots__')):
            ...
        elif isinstance(res, (numbers.Number, str, tuple, bytes, )):
            # handle primitive types
            ...

    return wrapper


def _copy_class_wrapper_attributes(base, wrapper):
    _MISSING = '_missing'
    for k in functools.WRAPPER_ASSIGNMENTS:
        # assign magic attributes like __module__, __qualname__, __doc__
        v = getattr(base, k, _MISSING)
        if v is not _MISSING:
            try:
                setattr(wrapper, k, v)
            except AttributeError:
                pass


def _argument_processor(arg, extra=None):
    # to convert argument before it is used in class or function
    # 1) auto-call get() to prevent type-converting in downstreaming functions
    if isinstance(arg, SerializableObject):
        arg = arg.get()
    # 2) see comments in `_freeze_list_and_dict`
    arg = _freeze_list_and_dict(arg)
    # 3) extra process, e.g., handle cases like ValueChoice
    if extra is not None:
        arg = extra(arg)
    return arg


def _freeze_list_and_dict(list_or_dict):
    # prevent the stored parameters to be mutated by inner class.
    # an example: https://github.com/microsoft/nni/issues/4329
    if isinstance(list_or_dict, list):
        return type(list_or_dict)(list_or_dict[:])
    if isinstance(list_or_dict, dict):
        # python dict is ordered by default now
        return type(list_or_dict)({k: v for k, v in list_or_dict.items()})
    return list_or_dict


def _formulate_arguments(func, args, kwargs, kw_only):
    # This is to formulate the arguments and make them well-formed.
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
            raise ValueError(f'Pickle too large when trying to dump {obj}. This might be caused by classes that are '
                             'not decorated by @nni.trace. Another option is to force bytes pickling and '
                             'try to raise pickle_size_limit.')
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
