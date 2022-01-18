import abc
import base64
import collections.abc
import copy
import functools
import inspect
import numbers
import types
import warnings
from io import IOBase
from typing import Any, Dict, List, Optional, TypeVar, Union

import cloudpickle  # use cloudpickle as backend for unserializable types and instances
import json_tricks  # use json_tricks as serializer backend

__all__ = ['trace', 'dump', 'load', 'PayloadTooLarge', 'Translatable', 'Traceable', 'is_traceable']


T = TypeVar('T')


class PayloadTooLarge(Exception):
    pass


class Traceable(abc.ABC):
    """
    A traceable object have copy and dict. Copy and mutate are used to copy the object for further mutations.
    Dict returns a TraceDictType to enable serialization.
    """
    @abc.abstractmethod
    def trace_copy(self) -> 'Traceable':
        """
        Perform a shallow copy.
        NOTE: NONE of the attributes will be preserved.
        This is the one that should be used when you want to "mutate" a serializable object.
        """
        ...

    @property
    @abc.abstractmethod
    def trace_symbol(self) -> Any:
        """
        Symbol object. Could be a class or a function.
        ``get_hybrid_cls_or_func_name`` and ``import_cls_or_func_from_hybrid_name`` is a pair to
        convert the symbol into a string and convert the string back to symbol.
        """
        ...

    @property
    @abc.abstractmethod
    def trace_args(self) -> List[Any]:
        """
        List of positional arguments passed to symbol. Usually empty if ``kw_only`` is true,
        in which case all the positional arguments are converted into keyword arguments.
        """
        ...

    @property
    @abc.abstractmethod
    def trace_kwargs(self) -> Dict[str, Any]:
        """
        Dict of keyword arguments.
        """
        ...


class Translatable(abc.ABC):
    """
    Inherit this class and implement ``translate`` when the wrapped class needs a different
    parameter from the wrapper class in its init function.
    """

    @abc.abstractmethod
    def _translate(self) -> Any:
        pass

    @staticmethod
    def _translate_argument(d: Any) -> Any:
        if isinstance(d, Translatable):
            return d._translate()
        return d


def is_traceable(obj: Any) -> bool:
    """
    Check whether an object is a traceable instance (not type).
    """
    return hasattr(obj, 'trace_copy') and \
        hasattr(obj, 'trace_symbol') and \
        hasattr(obj, 'trace_args') and \
        hasattr(obj, 'trace_kwargs') and \
        not inspect.isclass(obj)


class SerializableObject(Traceable):
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.
    """

    def __init__(self, symbol: T, args: List[Any], kwargs: Dict[str, Any], call_super: bool = False):
        # use dict to avoid conflicts with user's getattr and setattr
        self.__dict__['_nni_symbol'] = symbol
        self.__dict__['_nni_args'] = args
        self.__dict__['_nni_kwargs'] = kwargs
        self.__dict__['_nni_call_super'] = call_super

        if call_super:
            # call super means that the serializable object is by itself an object of the target class
            super().__init__(
                *[_argument_processor(arg) for arg in args],
                **{kw: _argument_processor(arg) for kw, arg in kwargs.items()}
            )

    def trace_copy(self) -> Union[T, 'SerializableObject']:
        return SerializableObject(
            self.trace_symbol,
            [copy.copy(arg) for arg in self.trace_args],
            {k: copy.copy(v) for k, v in self.trace_kwargs.items()},
        )

    @property
    def trace_symbol(self) -> Any:
        return self._get_nni_attr('symbol')

    @trace_symbol.setter
    def trace_symbol(self, symbol: Any) -> None:
        # for mutation purposes
        self.__dict__['_nni_symbol'] = symbol

    @property
    def trace_args(self) -> List[Any]:
        return self._get_nni_attr('args')

    @trace_args.setter
    def trace_args(self, args: List[Any]):
        self.__dict__['_nni_args'] = args

    @property
    def trace_kwargs(self) -> Dict[str, Any]:
        return self._get_nni_attr('kwargs')

    @trace_kwargs.setter
    def trace_kwargs(self, kwargs: Dict[str, Any]):
        self.__dict__['_nni_kwargs'] = kwargs

    def _get_nni_attr(self, name: str) -> Any:
        return self.__dict__['_nni_' + name]

    def __repr__(self):
        if self._get_nni_attr('call_super'):
            return super().__repr__()
        return 'SerializableObject(' + \
            ', '.join(['type=' + self._get_nni_attr('symbol').__name__] +
                      [repr(d) for d in self._get_nni_attr('args')] +
                      [k + '=' + repr(v) for k, v in self._get_nni_attr('kwargs').items()]) + \
            ')'


def inject_trace_info(obj: Any, symbol: T, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    # If an object is already created, this can be a fix so that the necessary info are re-injected into the object.

    def getter_factory(x):
        return lambda self: self.__dict__['_nni_' + x]

    def setter_factory(x):
        def setter(self, val):
            self.__dict__['_nni_' + x] = val

        return setter

    def trace_copy(self):
        return SerializableObject(
            self.trace_symbol,
            [copy.copy(arg) for arg in self.trace_args],
            {k: copy.copy(v) for k, v in self.trace_kwargs.items()},
        )

    attributes = {
        'trace_symbol': property(getter_factory('symbol'), setter_factory('symbol')),
        'trace_args': property(getter_factory('args'), setter_factory('args')),
        'trace_kwargs': property(getter_factory('kwargs'), setter_factory('kwargs')),
        'trace_copy': trace_copy
    }

    if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
        for name, method in attributes.items():
            setattr(obj.__class__, name, method)
    else:
        wrapper = type('wrapper', (Traceable, type(obj)), attributes)
        obj = wrapper(obj)  # pylint: disable=abstract-class-instantiated

    # make obj complying with the interface of traceable, though we cannot change its base class
    obj.__dict__.update(_nni_symbol=symbol, _nni_args=args, _nni_kwargs=kwargs)

    return obj


def trace(cls_or_func: T = None, *, kw_only: bool = True) -> Union[T, Traceable]:
    """
    Annotate a function or a class if you want to preserve where it comes from.
    This is usually used in the following scenarios:

    1) Care more about execution configuration rather than results, which is usually the case in AutoML. For example,
       you want to mutate the parameters of a function.
    2) Repeat execution is not an issue (e.g., reproducible, execution is fast without side effects).

    When a class/function is annotated, all the instances/calls will return a object as it normally will.
    Although the object might act like a normal object, it's actually a different object with NNI-specific properties.
    One exception is that if your function returns None, it will return an empty traceable object instead,
    which should raise your attention when you want to check whether the None ``is None``.

    When parameters of functions are received, it is first stored, and then a shallow copy will be passed to wrapped function/class.
    This is to prevent mutable objects gets modified in the wrapped function/class.
    When the function finished execution, we also record extra information about where this object comes from.
    That's why it's called "trace".
    When call ``nni.dump``, that information will be used, by default.

    If ``kw_only`` is true, try to convert all parameters into kwargs type. This is done by inspecting the argument
    list and types. This can be useful to extract semantics, but can be tricky in some corner cases.

    .. warning::

        Generators will be first expanded into a list, and the resulting list will be further passed into the wrapped function/class.
        This might hang when generators produce an infinite sequence. We might introduce an API to control this behavior in future.

    Example:

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass
    """

    def wrap(cls_or_func):
        # already annotated, do nothing
        if getattr(cls_or_func, '_traced', False):
            return cls_or_func
        if isinstance(cls_or_func, type):
            cls_or_func = _trace_cls(cls_or_func, kw_only)
        elif _is_function(cls_or_func):
            cls_or_func = _trace_func(cls_or_func, kw_only)
        else:
            raise TypeError(f'{cls_or_func} of type {type(cls_or_func)} is not supported to be traced. '
                            'File an issue at https://github.com/microsoft/nni/issues if you believe this is a mistake.')
        cls_or_func._traced = True
        return cls_or_func

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


def load(string: Optional[str] = None, *, fp: Optional[Any] = None, ignore_comments: bool = True, **json_tricks_kwargs) -> Any:
    """
    Load the string or from file, and convert it to a complex data structure.
    At least one of string or fp has to be not none.

    Parameters
    ----------
    string : str
        JSON string to parse. Can be set to none if fp is used.
    fp : str
        File path to load JSON from. Can be set to none if string is used.
    ignore_comments : bool
        Remove comments (starting with ``#`` or ``//``). Default is true.

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

    # to bypass a deprecation warning in json-tricks
    json_tricks_kwargs['ignore_comments'] = ignore_comments

    if string is not None:
        if isinstance(string, IOBase):
            raise TypeError(f'Expect a string, found a {string}. If you intend to use a file, use `nni.load(fp=file)`')
        return json_tricks.loads(string, obj_pairs_hooks=hooks, **json_tricks_kwargs)
    else:
        return json_tricks.load(fp, obj_pairs_hooks=hooks, **json_tricks_kwargs)


def _trace_cls(base, kw_only, call_super=True):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    class wrapper(SerializableObject, base):
        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            args, kwargs = _formulate_arguments(base.__init__, args, kwargs, kw_only, is_class_init=True)

            # calling serializable object init to initialize the full object
            super().__init__(symbol=base, args=args, kwargs=kwargs, call_super=call_super)

    _copy_class_wrapper_attributes(base, wrapper)

    return wrapper


def _trace_func(func, kw_only):
    @functools.wraps(func)
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
            # don't call super, makes no sense.
            # an empty serializable object is "none". Don't check it though.
            res = SerializableObject(func, args, kwargs, call_super=False)
        elif hasattr(res, '__class__') and hasattr(res, '__dict__'):
            # is a class, inject interface directly
            # need to be done before primitive types because there could be inheritance here.
            res = inject_trace_info(res, func, args, kwargs)
        elif isinstance(res, (collections.abc.Callable, types.ModuleType, IOBase)):
            raise TypeError(f'Try to add trace info to {res}, but functions and modules are not supported.')
        elif isinstance(res, (numbers.Number, collections.abc.Sequence, collections.abc.Set, collections.abc.Mapping)):
            # handle primitive types like int, str, set, dict, tuple
            # NOTE: simple types including none, bool, int, float, list, tuple, dict
            # will be directly captured by python json encoder
            # and thus not possible to restore the trace parameters after dump and reload.
            # this is a known limitation.
            res = inject_trace_info(res, func, args, kwargs)
        else:
            raise TypeError(f'Try to add trace info to {res}, but the type "{type(res)}" is unknown. '
                            'Please file an issue at https://github.com/microsoft/nni/issues')

        return res

    return wrapper


def _copy_class_wrapper_attributes(base, wrapper):
    _MISSING = '_missing'

    # assign magic attributes like __module__, __qualname__, __doc__
    for k in functools.WRAPPER_ASSIGNMENTS:
        v = getattr(base, k, _MISSING)
        if v is not _MISSING:
            try:
                setattr(wrapper, k, v)
            except AttributeError:
                pass

    wrapper.__wrapped__ = base


def _argument_processor(arg):
    # 1) translate
    # handle cases like ValueChoice
    # This is needed because sometimes the recorded arguments are meant to be different from what the wrapped object receives.
    arg = Translatable._translate_argument(arg)
    # 2) prevent the stored parameters to be mutated by wrapped class.
    # an example: https://github.com/microsoft/nni/issues/4329
    if isinstance(arg, (collections.abc.MutableMapping, collections.abc.MutableSequence, collections.abc.MutableSet)):
        arg = copy.copy(arg)
    return arg


def _formulate_single_argument(arg):
    # this is different from argument processor
    # it directly apply the transformation on the stored arguments

    # expand generator into list
    # Note that some types that are generator (such as range(10)) may not be identified as generator here.
    if isinstance(arg, types.GeneratorType):
        arg = list(arg)

    return arg


def _formulate_arguments(func, args, kwargs, kw_only, is_class_init=False):
    # This is to formulate the arguments and make them well-formed.
    if kw_only:
        # get arguments passed to a function, and save it as a dict
        argname_list = list(inspect.signature(func).parameters.keys())
        if is_class_init:
            argname_list = argname_list[1:]
        full_args = {}

        # match arguments with given arguments
        # args should be longer than given list, because args can be used in a kwargs way
        assert len(args) <= len(argname_list), f'Length of {args} is greater than length of {argname_list}.'
        for argname, value in zip(argname_list, args):
            full_args[argname] = value

        # use kwargs to override
        full_args.update(kwargs)

        args, kwargs = [], full_args

    args = [_formulate_single_argument(arg) for arg in args]
    kwargs = {k: _formulate_single_argument(arg) for k, arg in kwargs.items()}

    return list(args), kwargs


def _is_function(obj: Any) -> bool:
    # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
    return isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                            types.BuiltinMethodType))


def _import_cls_or_func_from_name(target: str) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


def _strip_trace_type(traceable: Any) -> Any:
    if getattr(traceable, '_traced', False):
        return traceable.__wrapped__
    return traceable


def _get_cls_or_func_name(cls_or_func: Any) -> str:
    module_name = cls_or_func.__module__
    if module_name == '__main__':
        raise ImportError('Cannot use a path to identify something from __main__.')
    full_name = module_name + '.' + cls_or_func.__name__

    try:
        imported = _import_cls_or_func_from_name(full_name)
        # ignores the differences in trace
        if _strip_trace_type(imported) != _strip_trace_type(cls_or_func):
            raise ImportError(f'Imported {imported} is not same as expected. The function might be dynamically created.')
    except ImportError:
        raise ImportError(f'Import {cls_or_func.__name__} from "{module_name}" failed.')

    return full_name


def get_hybrid_cls_or_func_name(cls_or_func: Any, pickle_size_limit: int = 4096) -> str:
    try:
        name = _get_cls_or_func_name(cls_or_func)
        # import success, use a path format
        return 'path:' + name
    except (ImportError, AttributeError):
        b = cloudpickle.dumps(cls_or_func)
        if len(b) > pickle_size_limit:
            raise ValueError(f'Pickle too large when trying to dump {cls_or_func}. '
                             'Please try to raise pickle_size_limit if you insist.')
        # fallback to cloudpickle
        return 'bytes:' + base64.b64encode(b).decode()


def import_cls_or_func_from_hybrid_name(s: str) -> Any:
    if s.startswith('bytes:'):
        b = base64.b64decode(s.split(':', 1)[-1])
        return cloudpickle.loads(b)
    if s.startswith('path:'):
        s = s.split(':', 1)[-1]
    return _import_cls_or_func_from_name(s)


def _json_tricks_func_or_cls_encode(cls_or_func: Any, primitives: bool = False, pickle_size_limit: int = 4096) -> str:
    if not isinstance(cls_or_func, type) and not _is_function(cls_or_func):
        # not a function or class, continue
        return cls_or_func

    return {
        '__nni_type__': get_hybrid_cls_or_func_name(cls_or_func, pickle_size_limit)
    }


def _json_tricks_func_or_cls_decode(s: Dict[str, Any]) -> Any:
    if isinstance(s, dict) and '__nni_type__' in s:
        s = s['__nni_type__']
        return import_cls_or_func_from_hybrid_name(s)
    return s


def _json_tricks_serializable_object_encode(obj: Any, primitives: bool = False, use_trace: bool = True) -> Dict[str, Any]:
    # Encodes a serializable object instance to json.

    # do nothing to instance that is not a serializable object and do not use trace
    if not use_trace or not is_traceable(obj):
        return obj

    if isinstance(obj.trace_symbol, property):
        # commonly made mistake when users forget to call the traced function/class.
        warnings.warn(f'The symbol of {obj} is found to be a property. Did you forget to create the instance with ``xx(...)``?')

    ret = {'__symbol__': get_hybrid_cls_or_func_name(obj.trace_symbol)}
    if obj.trace_args:
        ret['__args__'] = obj.trace_args
    if obj.trace_kwargs:
        ret['__kwargs__'] = obj.trace_kwargs
    return ret


def _json_tricks_serializable_object_decode(obj: Dict[str, Any]) -> Any:
    if isinstance(obj, dict) and '__symbol__' in obj:
        symbol = import_cls_or_func_from_hybrid_name(obj['__symbol__'])
        args = obj.get('__args__', [])
        kwargs = obj.get('__kwargs__', {})
        return trace(symbol)(*args, **kwargs)
    return obj


def _json_tricks_any_object_encode(obj: Any, primitives: bool = False, pickle_size_limit: int = 4096) -> Any:
    # We want to use this to replace the class instance encode in json-tricks.
    # Therefore the coverage should be roughly same.
    if isinstance(obj, list) or isinstance(obj, dict):
        return obj
    if hasattr(obj, '__class__') and (hasattr(obj, '__dict__') or hasattr(obj, '__slots__')):
        b = cloudpickle.dumps(obj)
        if len(b) > pickle_size_limit > 0:
            raise PayloadTooLarge(f'Pickle too large when trying to dump {obj}. This might be caused by classes that are '
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
