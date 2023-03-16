# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Can't have annotations on because PyTorch JIT doesn't support it.
# from __future__ import annotations

__all__ = [
    'recursive_freeze', 'MutableModule', 'ModelSpace', 'ParametrizedModule'
]

import copy
import itertools
import inspect
import logging
from typing import Any, Callable, Type, Iterable, Dict, Tuple, List, Optional, TypeVar

import torch
from torch import nn

from nni.mutable import (
    Mutable, LabeledMutable, Sample, SampleValidationError,
    ensure_frozen, frozen_context, label_scope, frozen
)
from nni.nas.space import BaseModelSpace, current_model, model_context

from nni.common.serializer import SerializableObject, _formulate_arguments, _copy_class_wrapper_attributes

_logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Mutable)


def recursive_freeze(module: nn.Module, sample: Dict[str, Any], include_root: bool = True) -> Tuple[nn.Module, bool]:
    """Recursively freeze all the mutables in a module.
    The sample is a dictionary of parameters.

    If ``include_root`` is true, it will call ``module.freeze(sample)`` if module is a mutable by itself.

    Returns
    -------
    A tuple of (frozen_module, replaced).
    If replaced is true, the child module has been replaced.
    """
    if include_root and isinstance(module, Mutable):
        return module.freeze(sample), True

    # Examine all the children.
    replaced_dict = {}
    for name, child_module in module.named_children():
        if isinstance(child_module, Mutable):
            new_module, replaced = child_module.freeze(sample), True
        else:
            new_module, replaced = recursive_freeze(child_module, sample)
        if replaced:
            replaced_dict[name] = new_module

    # If some children is replaced.
    if replaced_dict:
        module = copy.deepcopy(module)
        for name, new_module in replaced_dict.items():
            setattr(module, name, new_module)
        return module, True
    return module, False


class MutableModule(Mutable, nn.Module):
    """
    PyTorch module, but with uncertainties.

    This base class provides useful tools to handle search spaces built on top of PyTorch modules,
    including methods like :meth:`simplify`, :meth:`freeze`.

    :class:`MutableModule` can have dangling mutables registered on it via :meth:`add_mutable`.
    """

    def __new__(cls, *args, **kwargs):
        # The purpose of __new__ is to intercept module creation,
        # and create other types of modules when necessary.

        arch = current_model()
        if cls.should_invoke_fixed_module() and arch is not None:
            # If within a fixed_arch context, create the frozen module.
            # It must return a object with different type, or else infinite recursion will happen.
            return cls.create_fixed_module(arch, *args, **kwargs)  # type: ignore
        else:
            return super().__new__(cls)

    @classmethod
    def should_invoke_fixed_module(cls):
        """Call ``create_fixed_module()`` when fixed-arch context is detected.

        Typically this should be enabled. Otherwise the arch context might not be correctly handled.
        In cases where this flag is disabled, remember to detect arch context and
        manually freeze things in ``__init__``, or confirm that it's a composite module
        and nothing needs to be frozen.

        By default, it returns true when :meth:`create_fixed_module` is overridden.
        """
        return not getattr(cls.create_fixed_module, '_notimplemented', False)

    def add_mutable(self, mutable: T) -> T:
        """Register a mutable to this module.
        This is often used to add dangling variables that are not parameters of any :class:`ParametrizedModule`.

        If the mutable is also happens to be a submodule of type :class:`MutableModule`,
        it can be registered in the same way as PyTorch (i.e., ``self.xxx = mutable``). No need to add it again here.

        Examples
        --------
        In practice, this method is often used together with :func:`~nni.mutable.ensure_frozen`.

        >>> class MyModule(MutableModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         token_size = nni.choice('t', [4, 8, 16])        # Categorical variable here
        ...         self.add_mutable(token_size)                    # Register the mutable to this module.
        ...         real_token_size = ensure_frozen(token_size)     # Real number. 4 during dry run. 4, 8 or 16 during search.
        ...         self.token = nn.Parameter(torch.randn(real_token_size, 1))

        .. tip::

            Note that :func:`~nni.mutable.ensure_frozen` must be used under a :func:`~nni.mutable.frozen_context`.
            The easiest way to do so is to invoke it within initialization of a :class:`ModelSpace`.

        Warnings
        --------
        Arbitrary :meth:`add_mutable` is not supported for :class:`~nni.nas.space.GraphModelSpace`.
        """

        # NOTE:
        # This method can be potentially useful for every mutable. But I leave it here for the following reasons:
        # 1. If `_mutables` is to be put into base class, all subclasses need to call `super().__init__()`.
        # 2. Most of other mutables are "leaves". They do not have children. I don't agree that all mutables are "containers".
        # 3. To support nested mutables (like NestedCategorical), designs are needed (e.g., can constants be added via `add_mutable`?).
        # 4. The interface can be easily extended to base class if needed in future, without breaking backward compatibility.

        if isinstance(mutable, MutableModule):
            # If the mutable is also a module, it will be added automatically.
            return mutable

        if not isinstance(mutable, Mutable):
            raise TypeError(f'Expected Mutable, got {type(mutable)}: {mutable}')

        # Dynamically create it here to avoid an __init__ call.
        # Also avoiding dependencies for __init__ in ParametrizedModule.
        if not hasattr(self, '_mutables'):
            self._mutables: List[Mutable] = []

        self._mutables.append(mutable)

        # Disable dry run when:
        # 1. Inside a model_context; the `current_model()` needs to be read-only.
        # 2. ensure_frozen_strict is disabled. The `default()` can be called when `ensure_frozen()` is called.
        if current_model() is None:
            # We will "dry run" the mutable here, to obtain a default value for it.
            context = frozen_context.current()
            if context is None:
                if frozen._ENSURE_FROZEN_STRICT:
                    _logger.warning(
                        'No context found when `add_mutable()`. '
                        'You are probably adding a MutableModule outside the `__init__` of a ModelSpace. '
                        'This can possibly make the MutableModule untrackable and inconsistent: %s',
                        mutable
                    )
            else:
                context_before_keys = set(context.keys())

                try:
                    mutable.robust_default(context)
                except ValueError:
                    _logger.error('Error when trying to dry run the mutable: %r. It could be conflicted with current context: %s.',
                                  mutable, context)
                    raise

                frozen_context.update(
                    {key: value for key, value in context.items() if key not in context_before_keys}
                )

        return mutable

    @torch.jit.unused  # Must be marked as unused here. Otherwise JIT will look into Mutable and fail.
    @property
    def mutables(self) -> List[Mutable]:
        """Mutables that are dangling under this module.

        Normally this is all the mutables that are registered via :meth:`MutableModule.add_mutable`.
        """
        if not hasattr(self, '_mutables'):
            self._mutables: List[Mutable] = []

        return self._mutables

    # This is actually a classmethod, but decorated afterwards to assign `_notimplemented` attribute.
    # @classmethod
    def create_fixed_module(cls, sample: dict, *args, **kwargs) -> nn.Module:  # type: ignore
        """
        The classmethod is to create a brand new module with fixed architecture.

        The parameter ``sample`` is a dict with the exactly same format as ``sample`` in :meth:`freeze`.
        The difference is that when :meth:`create_fixed_module` is called,
        there is no :class:`MutableModule` instance created yet.
        Thus it can be useful to simplify the creation of a fixed module,
        by saving the cost of creating a :class:`MutableModule` instance and immediately :meth:`freeze` it.

        If automatic label generation (e.g., :func:`~nni.mutable.auto_label`) is used in ``__init__``,
        the same number of labels should be generated in this method.
        Otherwise it will mess up the global label counter, and potentially affect the label of successive modules.

        By default, this method has a not-implemented flag, and :meth:`should_invoke_fixed_module` will return ``False``
        based on this flag.
        """
        raise NotImplementedError('create_fixed_module() must be implemented when `custom_fixed_module_creation` is set to true.')

    create_fixed_module._notimplemented = True
    create_fixed_module = classmethod(create_fixed_module)  # type: ignore

    def check_contains(self, sample: Sample) -> Optional[SampleValidationError]:
        for mutable in self.mutables:
            exception = mutable.check_contains(sample)
            if exception is not None:
                return exception

        for name, module in self.named_mutable_descendants():
            exception = module.check_contains(sample)
            if exception is not None:
                exception.paths.append(name)
                return exception

        return None

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for mutable in self.mutables:
            yield from mutable.leaf_mutables(is_leaf)

        for module in self.mutable_descendants():
            yield from module.leaf_mutables(is_leaf)

    def mutable_descendants(self) -> Iterable['MutableModule']:
        """:meth:`named_mutable_descendants` without names."""
        for _, module in self.named_mutable_descendants():
            yield module

    def named_mutable_descendants(self) -> Iterable[Tuple[str, 'MutableModule']]:
        """Traverse the module subtree, find all descendants that are :class:`MutableModule`.

        - If a child module is :class:`MutableModule`, return it directly, and its subtree will be ignored.
        - If not, it will be recursively expanded, until :class:`MutableModule` is found.
        """
        def _iter(name: str, module: nn.Module) -> Iterable[Tuple[str, MutableModule]]:
            for subname, child in module.named_children():
                name_ = name + '.' + subname if name else subname
                if isinstance(child, MutableModule):
                    yield name_, child
                else:
                    yield from _iter(name_, child)

        yield from _iter('', self)

    def freeze(self, sample: Dict[str, Any]) -> nn.Module:
        """Return a frozen version of current mutable module.
        Some sub-modules can be possibly deep-copied.

        If mutables are added to the module via :meth:`add_mutable`, this method must be implemented.
        Otherwise, it will simply look at the children modules and freeze them recursively.

        :meth:`freeze` of subclass is encouraged to keep the original weights at best effort,
        but no guarantee is made, unless otherwise specified.
        """
        if self.mutables:
            raise NotImplementedError(
                'freeze() must be implemented when mutables have been registered to the module. '
                'A simple way to implement this is to recreate the current module after set the frozen context:\n\n'
                '    with nni.nas.space.model_context(sample):\n'
                '        return self.__class__(*args, **kwargs)\n\n'
                'Here, `args` and `kwargs` are the arguments to the constructor of the module.'
            )

        return recursive_freeze(self, sample, include_root=False)[0]

    def __repr__(self):
        return nn.Module.__repr__(self)


class TraceableMixin(Mutable):
    """
    This is actually another implementation of ``nni.trace()``.
    In this implementation, no decorator is needed. Users only need to inherit this class.
    ``is_traceable()`` check will still pass.

    For now, everything traced is on ``__init__``.
    Classes with customized ``__new__`` are not supported.
    """

    # Model space also needs to record init arguments, for recovering from another process.
    trace_args: tuple
    trace_kwargs: Dict[str, Any]

    # Useful in getting the signature of the original class __init__.
    _init_wrapped: Optional[Callable[..., None]] = None

    @torch.jit.ignore  # type: ignore
    def save_init_arguments(self, *args, **kwargs) -> None:
        self.trace_args = tuple(args)
        self.trace_kwargs = dict(kwargs)

    @torch.jit.ignore  # type: ignore
    def auto_save_init_arguments(self, *args, **kwargs) -> None:
        """Save init arguments into ``trace_args`` and ``trace_kwargs``.

        Skip when ``trace_args`` and ``trace_kwargs`` are already set,
        which could be possibly due to subclassing / inheritance.
        """
        # Stop saving when it's already saved, possibly in subclass's __init__.
        if not hasattr(self, 'trace_args') and not hasattr(self, 'trace_kwargs'):
            # If both super-class and sub-class calls this, subclass should first call this.
            init_fn = getattr(self.__class__, '_init_wrapped', self.__class__.__init__)
            args_, kwargs_ = _formulate_arguments(init_fn, args, kwargs, True, True)
            self.save_init_arguments(*args_, **kwargs_)

    @torch.jit.unused
    @property
    def trace_symbol(self):
        return self.__class__

    @torch.jit.unused
    @property
    def args(self) -> Dict[str, Any]:
        """All arguments that are used to construct the instance,
        including arguments passed to ``__init__``, as well as arguments with default value.

        Positional arguments are not supported.
        """
        if self.trace_args:
            raise RuntimeError('args is not available when positional argument is not empty.')
        rv = dict(self.trace_kwargs)
        # Add the arguments that are in signature and with default value.
        init_fn = getattr(self.__class__, '_init_wrapped', self.__class__.__init__)
        for param in inspect.signature(init_fn).parameters.values():
            if param.default is not param.empty and param.name not in rv:
                rv[param.name] = param.default
        return rv

    @torch.jit.ignore  # type: ignore
    def trace_copy(self):
        """Returns a different object here. All the model-specific details will be thrown away."""
        return SerializableObject(self.__class__, list(self.trace_args), self.trace_kwargs)


class ModelSpace(
    TraceableMixin,
    MutableModule,
    BaseModelSpace,
):
    """
    The base class for model search space based on PyTorch.
    The out-est module should inherit this class.

    Model space is written as PyTorch module for the convenience of writing code.
    It's not a real PyTorch model, and shouldn't be used as one for most cases.
    Most likely, the forward of :class:`ModelSpace` is a dry run of an arbitrary model in the model space.
    But since there is no guarantee on which model will be chosen, and the behavior is not well tested,
    it's only used for sanity check and tracing the space, and its semantics are not well-defined.

    Similarly for ``state_dict`` and ``load_state_dict``.
    Users should bear in mind that :class:`ModelSpace` is NOT a one-shot supernet,
    directly exporting its weights are unreliable and prone to error.
    Use :ref:`one-shot strategies <one-shot-nas>` to mutate the model space into a supernet for such needs.

    Mutables in model space **must all be labeled manually**, unless a label prefix is provided.
    Every model space can have a label prefix, which is used to provide a stable automatic label generation.
    For example, if the label prefix is ``model``, all the mutables initialized in a subclass of ModelSpace
    (in ``__init__`` function of itself and submodules, to be specific),
    will be automatically labeled with a prefix ``model/``.
    The label prefix can be manually specified upon definition of the class::

        class MyModelSpace(ModelSpace, label_prefix='backbone'):
            def __init__(self):
                super().__init__()

                self.choice = self.add_mutable(nni.choice('depth', [2, 3, 4]))
                print(self.choice.label)  # backbone/choice

    Notes
    -----
    The ``__init__`` implementation of :class:`ModelSpace` is in :func:`model_space_init_wrapper`.
    """

    # JIT can't parse label_scope. Deliberately ignore it to make JIT happy.
    # _label_scope: label_scope

    _label_prefix: Optional[str]
    """The label prefix of the model space."""

    def __init_subclass__(cls, disable_init_wrapper: bool = False, label_prefix: Optional[str] = None, **kwargs) -> None:
        # The init wrapper can be turned off in tricky cases.
        if not disable_init_wrapper:
            cls._init_wrapped = cls.__init__
            cls.__init__ = model_space_init_wrapper(cls.__init__)

        cls._label_prefix = label_prefix

    @classmethod
    def load_searched_model(cls, name: str, pretrained: bool = False, download: bool = False, progress: bool = True) -> nn.Module:
        """Load a pre-searched model with given name."""
        raise NotImplementedError('`load_searched_model` is not implemented, which means that no pre-searched model is available.')


class strict_label_scope(label_scope):
    """A strict label scope that raises error when label is not manually specified."""

    def next_label(self) -> str:
        raise ValueError('Label must be specified manually in NAS, or provide a `label_prefix` to the model space.')

    @property
    def path(self) -> Optional[List[str]]:
        """A strict label scope is only used for label checking. It shouldn't have its own name."""
        if self._path is None:
            return self._path
        return self._path[:-1]


def model_space_init_wrapper(original_init_fn: Callable[..., None]) -> Callable[..., None]:
    """Wrap the ``__init__`` inside a model namespace.

    This should only be wrapped on *subclasses* of :class:`ModelSpace`.
    It could possibly be wrapped multiple times when subclass of :class:`ModelSpace` is further subclassed.
    Currently, nested label namespace and dry run context will be created.
    """

    def init_with_context(self: ModelSpace, *args, **kwargs):
        arch = current_model()
        if arch is None:
            # If no arch is set, we are in dry run mode.
            # This is the only place where we set the dry run mode.
            # The context will record the sample generated during ``__init__``.
            # Some mutables use :func:`ensure_frozen` to freeze the sample.
            # The context is to make sure same label always return the same choice.
            self._frozen_context = frozen_context()
            with self._frozen_context:
                return original_init_fn(self, *args, **kwargs)
        else:
            # Already in context.
            self._frozen_context = frozen_context.top_context()
            return original_init_fn(self, *args, **kwargs)

    def new_init(self: ModelSpace, *args, **kwargs) -> None:
        # Save init arguments first. Skips if already saved.
        self.auto_save_init_arguments(*args, **kwargs)

        if not hasattr(self, '_label_scope'):
            if self._label_prefix is not None:
                self._label_scope = label_scope(self._label_prefix)
            else:
                self._label_scope = strict_label_scope('_unused_')  # the name is not used
        if hasattr(self, '_label_scope') and not self._label_scope.activated:  # type: ignore
            # Has a label scope but it's not activated. Create a "with".
            with self._label_scope:  # type: ignore
                return init_with_context(self, *args, **kwargs)
        else:
            return init_with_context(self, *args, **kwargs)

    return new_init


class ParametrizedModule(
    TraceableMixin,
    MutableModule,
):
    """
    Subclass of :class:`MutableModule` supports mutables as initialization parameters.

    One important feature of :class:`ParametrizedModule` is that
    it automatically freeze the mutable arguments passed to ``__init__``.
    This is for the convenience as well as compatibility with existing code::

        class MyModule(ParametrizedModule):
            def __init__(self, x):
                super().__init__()
                self.t = x   # Will be a fixed number, e.g., 3.

        MyModule(nni.choice('choice1', [1, 2, 3]))

    Note that the mutable arguments need to be directly posed as arguments to ``__init__``.
    They can't be hidden in a list or dict.

    If users want to make a 3rd-party module *parametrized*,
    it's recommended to do the following (taking ``nn.Conv2d`` as an example):

        >>> class ParametrizedConv2d(ParametrizedModule, nn.Conv2d, wraps=nn.Conv2d):
        ...     pass
        >>> conv = ParametrizedConv2d(3, nni.choice('out', [8, 16]))
        >>> conv
        >>> conv.out_channels
        8
        >>> conv.args['out_channels']
        Categorical([8, 16], label='out')
        >>> conv.freeze({'out': 16})
        Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))

    .. tip::

        The parametrized version of modules in ``torch.nn`` are already provided in ``nni.nas.nn.pytorch``.
        Every class is prefixed with ``Mutable``.
        For example, :class:`nni.nas.nn.pytorch.MutableConv2d`` is a parametrized version of ``torch.nn.Conv2d``.

    Attributes
    ----------
    args
        The arguments used to initialize the module.
        Since :class:`ParametrizedModule` will hijack the init arguments before passing to ``__init__``,
        this is the only recommended way to retrieve the original init arguments back.

    Warnings
    --------
    :class:`ParametrizedModule` can be nested.
    It's also possible to put arbitrary mutable modules inside a :class:`ParametrizedModule`.
    But be careful if the inner mutable modules are dependant on the parameters of :class:`ParametrizedModule`,
    because NNI can't handle cases where the mutables are a dynamically changing after initialization.
    For example, the following snippet is WRONG::

        class MyModule(ParametrizedModule):
            def __init__(self, x):
                if x == 0:
                    self.mutable = self.add_mutable(nni.choice('a', [1, 2, 3]))
                else:
                    self.mutable = self.add_mutable(nni.choice('b', [4, 5, 6]))

        module = MyModule(nni.choice('x', [0, 1]))
    """

    _nni_basic_unit: bool = True
    """This is only used in parsing graph. When ``_nni_basic_unit`` is true,
    the graph converter will consider the module as a primitive,
    and stop digging into the module to parse the inner modules.

    It's considered internal and should not be used by users.
    """

    _bound_type: Optional[Type] = None
    """The bounded class without :class:`ParametrizedModule`. Used to initialize a pure fixed instance.
    Set to none if :class:`ParametrizedModule` is the only superclass of the defined class."""

    @classmethod
    def should_invoke_fixed_module(cls) -> bool:
        return cls._bound_type is not None

    @torch.jit.ignore  # type: ignore
    def __init_subclass__(
        cls,
        disable_init_wrapper: bool = False,
        wraps: Optional[Type] = None,
        copy_wrapped: bool = False,
        pure_fixed_module: Optional[bool] = None,
        **kwargs
    ) -> None:
        # The init wrapper can be turned off in tricky cases.
        if not disable_init_wrapper:
            if wraps:
                cls.__wrapped__ = wraps  # type: ignore
                cls._init_wrapped = wraps.__init__
            else:
                cls._init_wrapped = cls.__init__
            cls.__init__ = parametrized_module_init_wrapper(cls.__init__)

        # Copy some attributes from the wrapped class, so that the wrapped class looks exactly like the initial one.
        # Useful in scenarios where module names are used to identify the class (e.g., graph space).
        if copy_wrapped:
            if wraps is None:
                raise ValueError('`wraps` should be specified when `copy_wrapped` is set to True.')
            _copy_class_wrapper_attributes(copy_wrapped, cls)

        if pure_fixed_module is None:
            pure_fixed_module = wraps is not None

        if pure_fixed_module:
            if wraps is None:
                raise ValueError('`wraps` should be specified when `pure_fixed_module` is set to True.')
            cls._bound_type = wraps

    @classmethod
    def create_fixed_module(cls, sample, *args, **kwargs) -> Any:
        assert cls._bound_type is not None, 'Cannot create fixed module for a class that is not bound to a fixed type.'
        args, kwargs = cls.freeze_init_arguments(sample, *args, **kwargs)
        with model_context(sample):  # A context should already exists. But it doesn't harm to create a new one.
            return cls._bound_type(*args, **kwargs)  # type: ignore  # pylint: disable=not-callable

    def freeze(self, sample: Dict[str, Any]) -> nn.Module:
        """Freeze all the mutable arguments in init.

        Note that a brand new module will be created, and all previous weights will be lost.
        Supernet must be created with one-shot strategies if you want to keep the weights.
        """
        args, kwargs = self.freeze_init_arguments(sample, *self.trace_args, **self.trace_kwargs)
        with model_context(sample):  # provide a context for nested mutable modules
            if self._bound_type is not None:
                return self._bound_type(*args, **kwargs)  # type: ignore  # pylint: disable=not-callable
            else:
                return self.__class__(*args, **kwargs)

    @staticmethod
    def freeze_init_arguments(sample: Optional[Sample], *args, **kwargs) -> Tuple[tuple, dict]:
        """Freeze the init arguments with the given context, and return the frozen arguments."""
        args_ = tuple(ensure_frozen(arg, sample=sample) for arg in args)
        kwargs_ = {kw: ensure_frozen(arg, sample=sample) for kw, arg in kwargs.items()}
        return args_, kwargs_

    def __repr__(self):
        params = []
        for value in self.trace_args:
            params.append(repr(value))
        for name, value in self.trace_kwargs.items():
            params.append(f'{name}={repr(value)}')
        return f'{self.__class__.__name__}({", ".join(params)})'


def parametrized_module_init_wrapper(original_init_fn: Callable[..., None]) -> Callable[..., None]:
    """Wrap the ``__init__`` for :class:`ParametrizedModule`.

    It returns a function, which first records all the parameters passed to ``__init__``,
    then ensures all the parameters are frozen, and finally calls the original ``__init__``.
    """

    def new_init(self: ParametrizedModule, *args, **kwargs):
        arch = current_model()
        if arch is not None:
            # This is the case where fixed-arch context exists,
            # yet ParametrizedModule doesn't have a bound type.
            args, kwargs = self.freeze_init_arguments(arch, *args, **kwargs)
            return original_init_fn(self, *args, **kwargs)

        # This is the case where fixed-arch context doesn't exist.
        self.auto_save_init_arguments(*args, **kwargs)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, Mutable):
                self.add_mutable(arg)
            else:
                _warn_if_nested_mutable(arg, self.__class__.__name__)
        # Sometimes, arguments will be hijacked to make the inner wrapped class happy.
        # For example Conv2d(choice([3, 5, 7])) should be Conv2d(3) instead,
        # because Conv2d doesn't recognize choice([3, 5, 7]).
        args, kwargs = self.freeze_init_arguments(None, *args, **kwargs)
        return original_init_fn(self, *args, **kwargs)

    return new_init


def _warn_if_nested_mutable(obj: Any, cls_name: str) -> None:
    # Warn for cases like MutableConv2d(kernel_size=(nni.choice([3, 5]), nni.choice([3, 5])))
    # This is not designed to be reliable, but only to be user-friendly.
    def _iter(o):
        if isinstance(o, Mutable):
            _logger.warning(f'Found a nested mutable {o} in parameter {obj} of class {cls_name}. '
                            'This is not recommended, because the mutable will not be tracked. '
                            'Please use MutableList, MutableDict instead, or write every options in a `nni.choice`.')
        else:
            if isinstance(o, (list, tuple, set)):
                for item in o:
                    _iter(item)
            elif isinstance(o, dict):
                for value in o.values():
                    _iter(value)

    _iter(obj)
