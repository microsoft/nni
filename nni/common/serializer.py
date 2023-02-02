# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import base64
import collections.abc
import copy
import functools
import inspect
import numbers
import os
import sys
import types
import warnings
from io import IOBase
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple, Union, cast

import cloudpickle  # use cloudpickle as backend for unserializable types and instances
import json_tricks  # use json_tricks as serializer backend

__all__ = ['trace', 'dump', 'load', 'PayloadTooLarge', 'Translatable', 'Traceable', 'is_traceable', 'is_wrapped_with_trace']


T = TypeVar('T')


class PayloadTooLarge(Exception):
    pass


class Traceable:
    """
    A traceable object have copy and dict. Copy and mutate are used to copy the object for further mutations.
    Dict returns a TraceDictType to enable serialization.
    """

    def trace_copy(self) -> 'Traceable':
        """
        Perform a shallow copy.
        NOTE: NONE of the attributes will be preserved.
        This is the one that should be used when you want to "mutate" a serializable object.
        """
        raise NotImplementedError()

    @property
    def trace_symbol(self) -> Any:
        """
        Symbol object. Could be a class or a function.
        ``get_hybrid_cls_or_func_name`` and ``import_cls_or_func_from_hybrid_name`` is a pair to
        convert the symbol into a string and convert the string back to symbol.
        """
        raise NotImplementedError()

    @property
    def trace_args(self) -> List[Any]:
        """
        List of positional arguments passed to symbol. Usually empty if ``kw_only`` is true,
        in which case all the positional arguments are converted into keyword arguments.
        """
        raise NotImplementedError()

    @property
    def trace_kwargs(self) -> Dict[str, Any]:
        """
        Dict of keyword arguments.
        """
        raise NotImplementedError()

    def get(self) -> Any:
        """
        Get the original object. Usually used together with ``trace_copy``.
        """
        raise NotImplementedError()


class Translatable(abc.ABC):
    """
    Inherit this class and implement ``translate`` when the wrapped class needs a different
    parameter from the wrapper class in its init function.

    .. deprecated:: 3.0
    """

    @abc.abstractmethod
    def _translate(self) -> Any:
        pass

    @staticmethod
    def _translate_argument(d: Any) -> Any:
        warnings.warn('Translatable is deprecated, please use `_unwrap_parameter` instead', DeprecationWarning)
        if isinstance(d, Translatable):
            return d._translate()
        return d


def is_traceable(obj: Any, must_be_instance: bool = False) -> bool:
    """
    Check whether an object is a traceable instance or type.

    Note that an object is traceable only means that it implements the "Traceable" interface,
    and the properties have been implemented. It doesn't necessary mean that its type is wrapped with trace,
    because the properties could be added **after** the instance has been created.

    If ``must_be_instance`` is set to true, the check returns false if ``obj`` is a type.
    """
    if must_be_instance and is_wrapped_with_trace(obj):
        return False
    return hasattr(obj, 'trace_copy') and \
        hasattr(obj, 'trace_symbol') and \
        hasattr(obj, 'trace_args') and \
        hasattr(obj, 'trace_kwargs')



def is_wrapped_with_trace(cls_or_func: Any) -> bool:
    """
    Check whether a function or class is already wrapped with ``@nni.trace``.
    If a class or function is already wrapped with trace, then the created object must be "traceable".
    """
    return getattr(cls_or_func, '_traced', False) and (
        not hasattr(cls_or_func, '__dict__') or  # in case it's a function
        '_traced' in cls_or_func.__dict__  # must be in this class, super-class traced doesn't count
    )


class SerializableObject(Traceable):  # should be (Generic[T], Traceable), but cloudpickle is unhappy with Generic.
    """
    Serializable object is a wrapper of existing python objects, that supports dump and load easily.
    Stores a symbol ``s`` and a dict of arguments ``args``, and the object can be restored with ``s(**args)``.

    :class:`SerializableObject` is not always a full object. Sometimes it could only
    contain necessary init arguments that are required to recreate another object.
    """

    def __init__(self, symbol: Type, args: List[Any], kwargs: Dict[str, Any], call_super: bool = False):
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

    def trace_copy(self) -> 'SerializableObject':
        return SerializableObject(
            self.trace_symbol,
            list(self.trace_args),
            dict(self.trace_kwargs),
        )

    def get(self, traceable: bool = True) -> Any:
        """Get the original instance. Reinstantiating if necessary.

        Parameters
        ----------
        traceable
            If true, the returned object is guaranteed to be still traceable.
        """
        if not self._get_nni_attr('call_super'):
            # Reinitialize
            if traceable:
                return trace(self.trace_symbol)(*self.trace_args, **self.trace_kwargs)
            else:
                return self.trace_symbol(*self.trace_args, **self.trace_kwargs)

        return self

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
        if ('_nni_' + name) not in self.__dict__:
            raise AttributeError(f'Attribute {name} not found in SerializableObject')
        return self.__dict__['_nni_' + name]

    def __repr__(self):
        if self._get_nni_attr('call_super'):
            return super().__repr__()
        return 'SerializableObject(' + \
            ', '.join(['type=' + self._get_nni_attr('symbol').__name__] +
                      [repr(d) for d in self._get_nni_attr('args')] +
                      [k + '=' + repr(v) for k, v in self._get_nni_attr('kwargs').items()]) + \
            ')'


def inject_trace_info(obj: Any, symbol: T, args: List[Any], kwargs: Dict[str, Any]) -> T:
    # If an object is already created, this can be a fix so that the necessary info are re-injected into the object.
    # Make obj complying with the interface of traceable, though we cannot change its base class.
    obj.__dict__.update(_nni_symbol=symbol, _nni_args=args, _nni_kwargs=kwargs)

    return obj


def _make_class_traceable(cls: T, create_wrapper: bool = False) -> T:
    # Make an already exist class traceable, without creating a new class.
    # Should be used together with `inject_trace_info`.

    def getter_factory(x):
        return lambda self: self.__dict__['_nni_' + x]

    def setter_factory(x):
        def setter(self, val):
            self.__dict__['_nni_' + x] = val

        return setter

    def trace_copy(self):
        return SerializableObject(
            self.trace_symbol,
            list(self.trace_args),
            dict(self.trace_kwargs),
        )

    def get(self):
        return self

    attributes = {
        'trace_symbol': property(getter_factory('symbol'), setter_factory('symbol')),
        'trace_args': property(getter_factory('args'), setter_factory('args')),
        'trace_kwargs': property(getter_factory('kwargs'), setter_factory('kwargs')),
        'trace_copy': trace_copy,
        'get': get,
    }

    if not create_wrapper:
        for name, method in attributes.items():
            setattr(cls, name, method)
        return cls
    else:
        # sometimes create_wrapper is mandatory, e.g., for built-in types like list/int.
        # but I don't want to check here because it's unreliable.
        wrapper = type('wrapper', (Traceable, cast(Type, cls)), attributes)
        return cast(T, wrapper)


def trace(cls_or_func: T = cast(T, None), *, kw_only: bool = True, inheritable: bool = False) -> T:
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

    When parameters of functions are received, it is first stored as ``trace_args`` and ``trace_kwargs``.
    ``_unwrap_parameter()`` will be invoked if it's defined on the parameter to do some transformations
    (e.g., :class:`~nni.mutable.Mutable` parameters can be transformed to fixed value to make the wrapped function happy).
    And then a shallow copy will be passed to wrapped function/class.
    This is to prevent mutable objects gets modified in the wrapped function/class.
    When the function finished execution, we also record extra information about where this object comes from.
    That's why it's called "trace".
    When call ``nni.dump``, that information will be used, by default.

    If ``kw_only`` is true, try to convert all parameters into kwargs type. This is done by inspecting the argument
    list and types. This can be useful to extract semantics, but can be tricky in some corner cases.
    Therefore, in some cases, some positional arguments will still be kept.

    If ``inheritable`` is true, the trace information from superclass will also be available in subclass.
    This however, will make the subclass un-trace-able. Note that this argument has no effect when tracing functions.

    Warnings
    --------
    Generators will be first expanded into a list, and the resulting list will be further passed into the wrapped function/class.
    This might hang when generators produce an infinite sequence. We might introduce an API to control this behavior in future.

    Examples
    --------

    .. code-block:: python

        @nni.trace
        def foo(bar):
            pass
    """

    # This is an internal flag to control the behavior of trace.
    # Useful in doc build and tests.
    # Might be changed in future.
    nni_trace_flag = os.environ.get('NNI_TRACE_FLAG', '')
    if nni_trace_flag.lower() == 'disable':
        return cast(T, cls_or_func)

    def wrap(cls_or_func):
        # already annotated, do nothing
        if is_wrapped_with_trace(cls_or_func):
            return cls_or_func
        if isinstance(cls_or_func, type):
            cls_or_func = _trace_cls(cls_or_func, kw_only, inheritable=inheritable)
        elif _is_function(cls_or_func):
            cls_or_func = _trace_func(cls_or_func, kw_only)
        else:
            raise TypeError(f'{cls_or_func} of type {type(cls_or_func)} is not supported to be traced. '
                            'File an issue at https://github.com/microsoft/nni/issues if you believe this is a mistake.')
        cls_or_func._traced = True
        return cls_or_func

    # if we're being called as @trace()
    if cls_or_func is None:
        return wrap  # type: ignore

    # if we are called without parentheses
    return wrap(cls_or_func)  # type: ignore


def dump(obj: Any, fp: Optional[Any] = None, *, use_trace: bool = True, pickle_size_limit: int = 4096,
         allow_nan: bool = True, **json_tricks_kwargs) -> str:
    """
    Convert a nested data structure to a json string. Save to file if fp is specified.
    Use json-tricks as main backend. For unhandled cases in json-tricks, use cloudpickle.
    The serializer is not designed for long-term storage use, but rather to copy data between processes.
    The format is also subject to change between NNI releases.

    It's recommended to use ``dump`` with ``trace``. The traced object can be stored with their traced arguments.
    For more complex objects, it will look for ``_dump`` and ``_load`` pair in the class.
    If not found, it will fallback to binary dump with cloudpickle.

    To compress the payload, please use :func:`dump_bytes`.

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

    if json_tricks_kwargs.get('compression') is not None:
        raise ValueError('If you meant to compress the dumped payload, please use `dump_bytes`.')
    result = _dump(
        obj=obj,
        fp=fp,
        use_trace=use_trace,
        pickle_size_limit=pickle_size_limit,
        allow_nan=allow_nan,
        **json_tricks_kwargs)
    return cast(str, result)


def dump_bytes(obj: Any, fp: Optional[Any] = None, *, compression: int = cast(int, None),
               use_trace: bool = True, pickle_size_limit: int = 4096,
               allow_nan: bool = True, **json_tricks_kwargs) -> bytes:
    """
    Same as :func:`dump`, but to comporess payload, with `compression <https://json-tricks.readthedocs.io/en/stable/#dump>`__.
    """
    if compression is None:
        raise ValueError('compression must be set.')
    result = _dump(
        obj=obj,
        fp=fp,
        compression=compression,
        use_trace=use_trace,
        pickle_size_limit=pickle_size_limit,
        allow_nan=allow_nan,
        **json_tricks_kwargs)
    return cast(bytes, result)


def _dump(*, obj: Any, fp: Optional[Any], use_trace: bool, pickle_size_limit: int,
          allow_nan: bool, **json_tricks_kwargs) -> Union[str, bytes]:
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
        _json_tricks_customize_encode,      # After serializable object
        functools.partial(_json_tricks_func_or_cls_encode, pickle_size_limit=pickle_size_limit),
        functools.partial(_json_tricks_any_object_encode, pickle_size_limit=pickle_size_limit),
    ]

    json_tricks_kwargs['allow_nan'] = allow_nan

    if fp is not None:
        return json_tricks.dump(obj, fp, obj_encoders=encoders, **json_tricks_kwargs)
    else:
        return json_tricks.dumps(obj, obj_encoders=encoders, **json_tricks_kwargs)


def load(string: Optional[str] = None, *, fp: Optional[Any] = None,
         preserve_order: bool = False, ignore_comments: bool = True, **json_tricks_kwargs) -> Any:
    """
    Load the string or from file, and convert it to a complex data structure.
    At least one of string or fp has to be not none.

    Parameters
    ----------
    string : str
        JSON string to parse. Can be set to none if fp is used.
    fp : str
        File path to load JSON from. Can be set to none if string is used.
    preserve_order : bool
        `json_tricks parameter <https://json-tricks.readthedocs.io/en/latest/#order>`_
        to use ``OrderedDict`` instead of ``dict``.
        The order is in fact always preserved even when this is False.
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
        _json_tricks_customize_decode,
        _json_tricks_func_or_cls_decode,
        _json_tricks_any_object_decode
    ]

    # there was an issue that the user code does not accept ordered dict, and 3.7+ dict has guaranteed order
    json_tricks_kwargs['preserve_order'] = preserve_order
    # to bypass a deprecation warning in json-tricks
    json_tricks_kwargs['ignore_comments'] = ignore_comments

    if string is not None:
        if isinstance(string, IOBase):
            raise TypeError(f'Expect a string, found a {string}. If you intend to use a file, use `nni.load(fp=file)`')
        return json_tricks.loads(string, obj_pairs_hooks=hooks, **json_tricks_kwargs)
    else:
        return json_tricks.load(fp, obj_pairs_hooks=hooks, **json_tricks_kwargs)


def _trace_cls(base, kw_only, call_super=True, inheritable=False):
    # the implementation to trace a class is to store a copy of init arguments
    # this won't support class that defines a customized new but should work for most cases

    if sys.platform != 'linux':
        if not call_super:
            raise ValueError("'call_super' is mandatory to be set true on non-linux platform")

        try:
            # In non-linux envs, dynamically creating new classes doesn't work with pickle.
            # We have to replace the ``__init__`` with a new ``__init__``.
            # This, however, causes side-effects where the replacement is not intended.
            # This also doesn't work built-in types (e.g., OrderedDict), and the replacement
            # won't be effective any more if ``nni.trace`` is called in-place (e.g., ``nni.trace(nn.Conv2d)(...)``).
            original_init = base.__init__

            # Makes the new init have the exact same signature as the old one,
            # so as to make pytorch-lightning happy.
            # https://github.com/PyTorchLightning/pytorch-lightning/blob/4cc05b2cf98e49168a5f5dc265647d75d1d3aae9/pytorch_lightning/utilities/parsing.py#L143
            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                args, kwargs = _formulate_arguments(original_init, args, kwargs, kw_only, is_class_init=True)
                original_init(
                    self,
                    *[_argument_processor(arg) for arg in args],
                    **{kw: _argument_processor(arg) for kw, arg in kwargs.items()}
                )
                inject_trace_info(self, base, args, kwargs)

            base.__init__ = new_init

            base = _make_class_traceable(base)
            return base

        except TypeError:
            warnings.warn("In-place __init__ replacement failed in `@nni.trace`, probably because the type is a built-in/extension type, "
                          "and it's __init__ can't be replaced. `@nni.trace` is now falling back to the 'inheritance' approach. "
                          "However, this could cause issues when using pickle. See https://github.com/microsoft/nni/issues/4434",
                          RuntimeWarning)

    # This is trying to solve the case where superclass and subclass are both decorated with @nni.trace.
    # We use a metaclass to "unwrap" the superclass.
    # However, this doesn't work if:
    # 1. Base class already has a customized metaclass. We will raise error in that class.
    # 2. SerializableObject in ancester (instead of parent). I think this case is rare and I didn't handle this case yet. FIXME
    if type(base) is type and not inheritable:
        metaclass = _unwrap_metaclass
    else:
        metaclass = type
        if SerializableObject in inspect.getmro(base):
            raise TypeError(f"{base} has a superclass already decorated with trace, and it's using a customized metaclass {type(base)}. "
                            "Please either use the default metaclass, or remove trace from the super-class.")

    class wrapper(SerializableObject, base, metaclass=metaclass):  # type: ignore
        def __init__(self, *args, **kwargs):
            # store a copy of initial parameters
            args, kwargs = _formulate_arguments(base.__init__, args, kwargs, kw_only, is_class_init=True)

            try:
                # calling serializable object init to initialize the full object
                super().__init__(symbol=base, args=args, kwargs=kwargs, call_super=call_super)
            except RecursionError as e:
                warnings.warn(
                    'Recursion error detected in initialization of wrapped object. '
                    'Did you use `super(MyClass, self).__init__()` rather than `super().__init__()`? '
                    'Please use `super().__init__()` and try again. '
                    f'Original error: {e}',
                    RuntimeWarning
                )
                raise

        def __reduce__(self):
            # The issue that decorator and pickler doesn't play well together is well known.
            # The workaround solution is to use a fool class (_pickling_object) which pretends to be the pickled object.
            # We then put the original type, as well as args and kwargs in its `__new__` argument.
            # I suspect that their could still be problems when things get complex,
            # e.g., the wrapped class has a custom pickling (`__reduce__``) or `__new__`.
            # But it can't be worse because the previous pickle doesn't work at all.
            #
            # Linked issue: https://github.com/microsoft/nni/issues/4434
            # SO: https://stackoverflow.com/questions/52185507/pickle-and-decorated-classes-picklingerror-not-the-same-object

            # Store the inner class. The wrapped class couldn't be properly pickled.
            type_ = cloudpickle.dumps(type(self).__wrapped__)

            # in case they have customized ``__getstate__``.
            if hasattr(self, '__getstate__'):
                obj_ = self.__getstate__()
            else:
                obj_ = self.__dict__

            # Pickle can't handle type objects.
            if '_nni_symbol' in obj_:
                obj_ = dict(obj_)  # copy the object to keep the original symbol unchanged
                obj_['_nni_symbol'] = cloudpickle.dumps(obj_['_nni_symbol'])

            return _pickling_object, (type_, kw_only, obj_)

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
            if not getattr(type(res), '_traced', False):
                _make_class_traceable(type(res), False)  # in-place
            res = inject_trace_info(res, func, args, kwargs)
        elif isinstance(res, (collections.abc.Callable, types.ModuleType, IOBase)):
            raise TypeError(f'Try to add trace info to {res}, but functions and modules are not supported.')
        elif isinstance(res, (numbers.Number, collections.abc.Sequence, collections.abc.Set, collections.abc.Mapping)):
            # handle primitive types like int, str, set, dict, tuple
            # NOTE: simple types including none, bool, int, float, list, tuple, dict
            # will be directly captured by python json encoder
            # and thus not possible to restore the trace parameters after dump and reload.
            # this is a known limitation.
            new_type = _make_class_traceable(type(res), True)
            # re-creating the object
            res = new_type(res)  # type: ignore
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


class _unwrap_metaclass(type):
    # When a subclass is created, it detects whether the super-class is already annotated with @nni.trace.
    # If yes, it gets the ``__wrapped__`` inner class, so that it doesn't inherit SerializableObject twice.
    # Note that this doesn't work when metaclass is already defined (such as ABCMeta). We give up in that case.

    def __new__(cls, name, bases, dct):
        bases = tuple([getattr(base, '__wrapped__', base) for base in bases])
        return super().__new__(cls, name, cast(Tuple[type, ...], bases), dct)

    # Using a customized "bases" breaks default isinstance and issubclass.
    # We recover this by overriding the subclass and isinstance behavior, which conerns wrapped class only.
    def __subclasscheck__(cls, subclass):
        inner_cls = getattr(cls, '__wrapped__', cls)
        return inner_cls in inspect.getmro(subclass)

    def __instancecheck__(cls, instance):
        inner_cls = getattr(cls, '__wrapped__', cls)
        return inner_cls in inspect.getmro(type(instance))


class _pickling_object:
    # Need `cloudpickle.load` on the callable because the callable is pickled with cloudpickle.
    # Used in `_trace_cls`.

    def __new__(cls, type_, kw_only, data):
        type_ = _wrapped_cloudpickle_loads(type_)
        # Restore the trace type
        type_ = _trace_cls(type_, kw_only)

        # restore type
        if '_nni_symbol' in data:
            data['_nni_symbol'] = _wrapped_cloudpickle_loads(data['_nni_symbol'])

        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        obj = type_.__new__(type_)
        if hasattr(obj, '__setstate__'):
            obj.__setstate__(data)
        else:
            obj.__dict__.update(data)
        return obj


def _argument_processor(arg):
    # 1) translate
    # handle cases like ValueChoice
    # This is needed because sometimes the recorded arguments are meant to be different from what the wrapped object receives.

    if hasattr(arg, '_unwrap_parameter'):
        arg = arg._unwrap_parameter()

    # deprecated
    if isinstance(arg, Translatable):
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
        # Match arguments with given arguments, so that we can use keyword arguments as much as possible.
        # Mutators don't like positional arguments. Positional arguments might not supply enough information.

        # get arguments passed to a function, and save it as a dict
        insp_parameters = inspect.signature(func).parameters
        argname_list = list(insp_parameters.keys())
        if is_class_init:
            argname_list = argname_list[1:]
        positional_args = []
        keyword_args = {}

        # According to https://docs.python.org/3/library/inspect.html#inspect.Parameter, there are five kinds of parameters
        # in Python. We only try to handle POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD here.

        # Example:
        # For foo(a, b, *c, **d), a and b and c should be kept.
        # For foo(a, b, /, d), a and b should be kept.

        for i, value in enumerate(args):
            if i >= len(argname_list):
                raise ValueError(f'{func} receives extra argument: {value}.')

            argname = argname_list[i]
            if insp_parameters[argname].kind == inspect.Parameter.POSITIONAL_ONLY:
                # positional only. have to be kept.
                positional_args.append(value)

            elif insp_parameters[argname].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # this should be the most common case
                keyword_args[argname] = value

            elif insp_parameters[argname].kind == inspect.Parameter.VAR_POSITIONAL:
                # Any previous preprocessing might be wrong. Clean them all.
                # Any parameters that appear before a VAR_POSITIONAL should be kept positional.
                # Otherwise, VAR_POSITIONAL might not work.
                # For the cases I've tested, any parameters that appear after a VAR_POSITIONAL are considered keyword only.
                # But, if args is not long enough for VAR_POSITIONAL to be encountered, they should be handled by other if-branches.
                positional_args = args
                keyword_args = {}
                break

            else:
                # kind has to be one of `KEYWORD_ONLY` and `VAR_KEYWORD`
                raise ValueError(f'{func} receives positional argument: {value}, but the parameter type is found to be keyword only.')

        # use kwargs to override
        keyword_args.update(kwargs)

        if positional_args:
            # Raise a warning if some arguments are not convertible to keyword arguments.
            warnings.warn(f'Found positional arguments {positional_args} should processing parameters of {func}. '
                          'We recommend always using keyword arguments to specify parameters. '
                          'For example: `nn.LSTM(input_size=2, hidden_size=2)` instead of `nn.LSTM(2, 2)`.')

    else:
        # keep them unprocessed
        positional_args, keyword_args = args, kwargs

    # do some extra conversions to the arguments.
    positional_args = [_formulate_single_argument(arg) for arg in positional_args]
    keyword_args = {k: _formulate_single_argument(arg) for k, arg in keyword_args.items()}

    return positional_args, keyword_args


def _is_function(obj: Any) -> bool:
    # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
    return isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                            types.BuiltinMethodType)) and obj is not None


def _import_cls_or_func_from_name(target: str) -> Any:
    if target is None:
        return None
    path, identifier = target.rsplit('.', 1)
    module = __import__(path, globals(), locals(), [identifier])
    return getattr(module, identifier)


def _strip_trace_type(traceable: Any) -> Any:
    if getattr(traceable, '_traced', False):
        # sometimes, ``__wrapped__`` could be unavailable (e.g., with `inject_trace_info`)
        # need to have a default value
        return getattr(traceable, '__wrapped__', traceable)
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
    """Pickle a class or function object to a string.

    It will first try to picklize the object with an importable path.
    If that doesn't work out, it fallbacks to cloudpickle.
    """
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
        return _wrapped_cloudpickle_loads(b)
    if s.startswith('path:'):
        s = s.split(':', 1)[-1]
    return _import_cls_or_func_from_name(s)


def _json_tricks_func_or_cls_encode(cls_or_func: Any, primitives: bool = False, pickle_size_limit: int = 4096) -> Dict[str, str]:
    if not isinstance(cls_or_func, type) and not _is_function(cls_or_func):
        # not a function or class, continue
        return cls_or_func

    return {
        '__nni_type__': get_hybrid_cls_or_func_name(cls_or_func, pickle_size_limit)
    }


def _json_tricks_func_or_cls_decode(s: Dict[str, Any]) -> Any:
    if isinstance(s, dict) and '__nni_type__' in s:
        return import_cls_or_func_from_hybrid_name(s['__nni_type__'])
    return s


def _json_tricks_serializable_object_encode(obj: Any, primitives: bool = False, use_trace: bool = True) -> Dict[str, Any]:
    # Encodes a serializable object instance to json.

    # do nothing to instance that is not a serializable object and do not use trace
    if not (use_trace and is_traceable(obj, must_be_instance=True)):
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


def _json_tricks_customize_encode(obj: Any, primitives: bool = False) -> Any:
    # Dealing with classes with dump and load pair.
    if hasattr(obj, '_dump') and hasattr(obj, '_load'):
        dump_res = obj._dump()
        if not isinstance(dump_res, dict):
            raise ValueError(f'Customized object {obj} must return a dict when calling _dump().')
        if '__instance_type__' in dump_res:
            raise ValueError(f'Customized object {obj} cannot have key "__instance_type__" in the dict returned by _dump().')
        dump_res['__instance_type__'] = get_hybrid_cls_or_func_name(type(obj))
        return dump_res
    return obj


def _json_tricks_customize_decode(obj: Dict[str, Any]) -> Any:
    if isinstance(obj, dict) and '__instance_type__' in obj:
        cls = import_cls_or_func_from_hybrid_name(obj['__instance_type__'])
        if not hasattr(cls, '_load'):
            raise ValueError(f'Customized object {cls} must have a static method _load() to load from a dict.')
        kwargs = obj.copy()
        kwargs.pop('__instance_type__')
        return cls._load(**kwargs)
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
        b = base64.b64decode(obj['__nni_obj__'])
        return _wrapped_cloudpickle_loads(b)
    return obj


def _wrapped_cloudpickle_loads(b: bytes) -> Any:
    try:
        return cloudpickle.loads(b)
    except TypeError:
        warnings.warn('TypeError encountered during deserializing object. This could be caused by '
                      'inconsistency between Python versions where dump and load happens.')
        raise
