# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'shape_inference', 'ShapeTensor', 'MutableShape', 'submodule_input_output_shapes', 'extract_shape_info'
]

import functools
import logging
import traceback
from typing import Union, Tuple, Callable, Iterable, Any, NamedTuple, Dict, cast, overload

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from nni.mutable import MutableExpression, Mutable, LabeledMutable, Sample, SampleValidationError
from nni.mutable.mutable import _mutable_equal

from .misc import is_leaf_module, argument_in_spec


IntExpression = Union[MutableExpression[int], int]


class _ModuleName(NamedTuple):
    id: int
    name: str
    type_name: str
    has_formula: bool


_current_module_names: list[_ModuleName] = []
"""Trace current module name to give meaningful warning messages in ``__torch_dispatch__``."""

_logger = logging.getLogger(__name__)


def extract_shape_info(x):
    """Extract shape information from tensors (and potentially complex data structures) returned by module forward."""
    from torch.utils._pytree import tree_map

    def _extract(x):
        if isinstance(x, ShapeTensor):
            return x.real_shape
        return x

    return tree_map(_extract, x)


def switch_case_shape_info(indicator: Any, branches: dict[str, Any]) -> Any:
    """Combine shape information obtained from multiple branches together.

    The return type will look at one of the branches,
    ``switch_case`` will deeply merged into every shape dimension which is different across branches.

    Examples
    --------
    >>> switch_case_shape_info(nni.choice('x', [1, 2, 3]), {
    ...     1: MutableShape(2, 3),
    ...     2: MutableShape(4, 5),
    ...     3: MutableShape(6, 7),
    ... })
    MutableShape(
        switch_case(Categorical([1, 2, 3], label='x'), {1: 2, 2: 4, 3: 6}), 
        switch_case(Categorical([1, 2, 3], label='x'), {1: 3, 2: 5, 3: 7})
    )
    """
    from torch.utils._pytree import tree_flatten, tree_unflatten

    cases, shapes_flattened, spec = [], [], None
    for case, item in branches.items():
        s_flattened, cur_spec = tree_flatten(item)
        if spec is None:
            spec = cur_spec
        elif spec != cur_spec:
            raise ValueError('Shape information must have the same structure, '
                             f'but got {spec} vs {cur_spec}')
        if shapes_flattened and len(shapes_flattened[0]) != len(s_flattened):
            raise ValueError('Shape information must have the same number of elements, '
                             f'but got {shapes_flattened[0]} vs {s_flattened}')
        cases.append(case)
        shapes_flattened.append(s_flattened)

    assert spec is not None, 'branches can not be empty'

    shapes_combined = []

    # This loop iterate over shapes of multiple tensors.
    for i in range(len(shapes_flattened[0])):
        # Shapes at this position are from different branches.
        to_merge = [s[i] for s in shapes_flattened]

        if not isinstance(to_merge[0], MutableShape):
            # Do nothing for non-MutableShape.
            shapes_combined.append(to_merge[0])
        else:
            # We make sure every shape to merge has the same dimension.
            assert all(len(shape) == len(to_merge[0]) for shape in to_merge)
            # This loop iterate over elements of a shape.
            # Create every element of the output shape separately, so that the return value is a MutableShape.
            elements = []
            for j in range(len(to_merge[0])):
                to_merge_j = [s[j] for s in to_merge]
                if all(_mutable_equal(s, to_merge_j[0]) for s in to_merge_j):
                    # If all elements are the same, we can just use the first one.
                    # This is to avoid creating an unnecessary MutableExpression.
                    elements.append(to_merge_j[0])
                else:
                    elements.append(MutableExpression.switch_case(
                        indicator, {case: s for case, s in zip(cases, to_merge_j)}
                    ))
            shapes_combined.append(MutableShape(*elements))

    return tree_unflatten(shapes_combined, spec)


def _assign_shape_info(tensors, shapes):
    """Assign shape information returned by formula, to tensors returned by module forward."""
    from torch.utils._pytree import tree_flatten, tree_unflatten

    tensors_flatten, tree_spec_tensor = tree_flatten(tensors)
    shapes_flatten, tree_spec_shapes = tree_flatten(shapes)
    if tree_spec_tensor != tree_spec_shapes:
        raise ValueError('Shape formula must return the same structure as the output, '
                         f'but got {tree_spec_tensor} vs {tree_spec_shapes}')
    if len(tensors_flatten) != len(shapes_flatten):
        raise ValueError('Shape formula must return the same number of elements as the number of outputs, '
                         f'but got {tensors_flatten} vs {shapes_flatten}')
    for i, s in enumerate(shapes_flatten):
        if isinstance(tensors_flatten[i], ShapeTensor):
            if isinstance(s, MutableShape):
                tensors_flatten[i].real_shape = s
            elif isinstance(s, ShapeTensor):
                tensors_flatten[i] = s
            else:
                raise ValueError(f'Expected MutableShape or ShapeTensor in return value of shape formula, but got {type(s)}')

    return tree_unflatten(tensors_flatten, tree_spec_tensor)


class MutableShape(Mutable):
    """This is very similar to ``torch.Size``, but carries a symbolic expression.

    The shape behaves like a tuple, and it can't inherit tuple.
    Otherwise it would be flattened when passed to :func:`_assign_shape_info`.
    """

    def __init__(self, *shape: tuple[IntExpression, ...] | IntExpression):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, MutableShape)):
            self._shape = cast(Tuple[IntExpression, ...], shape[0])
        else:
            self._shape = cast(Tuple[IntExpression, ...], tuple(shape))
        for s in self._shape:
            if not isinstance(s, int) and not isinstance(s, MutableExpression):
                raise TypeError(f'Invalid shape element: {s} in {self._shape}')
            if isinstance(s, MutableExpression) and not isinstance(s.default(), int):
                raise TypeError(f'Only int or int expression is allowed in shape, but got {s!r}')

    def freeze(self, sample: Sample) -> torch.Size:
        return torch.Size(s.freeze(sample) if isinstance(s, Mutable) else s for s in self._shape)

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        for i, s in enumerate(self._shape):
            if isinstance(s, Mutable):
                err = s.check_contains(sample)
                if err:
                    err.paths.append(f'dim: {i}')
                    return err

    def leaf_mutables(self, is_leaf: Callable[[Mutable], bool]) -> Iterable[LabeledMutable]:
        for s in self._shape:
            if isinstance(s, Mutable):
                yield from s.leaf_mutables(is_leaf)

    def numel(self):
        return functools.reduce(lambda x, y: x * y, self._shape, 1)

    def is_mutable(self):
        """Check if the shape contains any mutable element."""
        return any(isinstance(s, Mutable) for s in self._shape)

    @overload
    def __getitem__(self, index: int) -> IntExpression:
        ...

    @overload
    def __getitem__(self, index: slice) -> MutableShape:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MutableShape(self._shape[index])
        else:
            return self._shape[index]

    def __repr__(self):
        return f'MutableShape{self._shape}'

    def __iter__(self):
        return iter(self._shape)

    def __len__(self):
        return len(self._shape)

    def __eq__(self, other):
        return isinstance(other, MutableShape) and _mutable_equal(self._shape, other._shape)

    def __ne__(self, other):
        return not self.__eq__(other)


Formula = Callable[..., Any]


class IntProxy(int):
    """Works like a int, but carries an expression underneath.

    Doing ``+``, ``-``, ``*``, ``/`` and ``//`` on this proxy will return a new proxy.

    Reference: https://stackoverflow.com/questions/3238350/subclassing-int-in-python
    """
    expression: IntExpression

    @staticmethod
    def unwrap(proxy: IntProxy | int) -> IntExpression:
        if isinstance(proxy, IntProxy):
            return proxy.expression
        return proxy

    def __new__(cls, value: int, expression: IntExpression):
        obj = super().__new__(cls, value)
        obj.expression = expression
        return obj

    def __add__(self, other):
        obj = super(IntProxy, self).__add__(other)
        return self.__class__(obj, self.expression + IntProxy.unwrap(other))

    def __radd__(self, other):
        obj = super(IntProxy, self).__radd__(other)
        return self.__class__(obj, IntProxy.unwrap(other) + self.expression)

    def __sub__(self, other):
        obj = super(IntProxy, self).__sub__(other)
        return self.__class__(obj, self.expression - IntProxy.unwrap(other))

    def __rsub__(self, other):
        obj = super(IntProxy, self).__rsub__(other)
        return self.__class__(obj, IntProxy.unwrap(other) - self.expression)

    def __mul__(self, other):
        obj = super(IntProxy, self).__mul__(other)
        return self.__class__(obj, self.expression * IntProxy.unwrap(other))

    def __rmul__(self, other):
        obj = super(IntProxy, self).__rmul__(other)
        return self.__class__(obj, IntProxy.unwrap(other) * self.expression)

    def __truediv__(self, other):
        raise RuntimeError('Division (`/`) on shape dimensions is not supported. Use `//` instead.')

    def __rtruediv__(self, other):
        raise RuntimeError('Division (`/`) on shape dimensions is not supported. Use `//` instead.')

    def __floordiv__(self, other):
        obj = super(IntProxy, self).__floordiv__(other)
        return self.__class__(obj, self.expression // IntProxy.unwrap(other))

    def __rfloordiv__(self, other):
        obj = super(IntProxy, self).__rfloordiv__(other)
        return self.__class__(obj, IntProxy.unwrap(other) // self.expression)

    def __repr__(self):
        return f'IntProxy({int(self)}, {self.expression})'


class ShapeTensor(torch.Tensor):
    """A tensor with mutable shape information.
    Inspired by `Ideal FLOPS counter <https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505>`__.
    """

    elem: torch.Tensor
    """A tensor that is used for dry run the model.
    Since the tensor must have a fixed shape, the shape of the tensor is not accurate.
    Mutable shapes are stored in ``real_shape``.
    """

    real_shape: MutableShape | None
    """If not null, this ``real_shape`` must be accurate and precisely what the tensor is supposed to be."""

    __slots__ = ['elem', 'real_shape']

    @staticmethod
    def __new__(cls, elem: torch.Tensor, shape: bool | tuple = False):
        assert isinstance(elem, torch.Tensor) and not isinstance(elem, ShapeTensor)
        # The wrapping tensor (ShapeTensor) shouldn't hold any memory for the class in question,
        # but it should still advertise the same device as before.
        r: ShapeTensor = torch.Tensor._make_wrapper_subclass(  # type: ignore
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        if shape is True:
            r.real_shape = MutableShape(*elem.size())
        elif shape is False:
            r.real_shape = None
        else:
            r.real_shape = MutableShape(shape)
        return r

    def __repr__(self):
        return f"ShapeTensor({list(self.real_shape) if self.real_shape is not None else 'unknown'})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_map
        from .shape_formula import find_shape_inference_formula

        kwargs = cast(Dict[str, Any], kwargs)

        def unwrap(e):
            return e.elem if isinstance(e, ShapeTensor) else e

        def wrap(e):
            return ShapeTensor(e) if isinstance(e, torch.Tensor) and not isinstance(e, ShapeTensor) else e

        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        rs = tree_map(wrap, rs)

        if _current_module_names and _current_module_names[-1].has_formula:
            # The closest module has a shape formula, so we don't need to infer the shape.
            return rs

        formula = find_shape_inference_formula(func)

        def format_module_stack():
            current_modules_info = [
                f'  - {m.name!r} (type: {m.type_name}, ' + ('NO' if not m.has_formula else 'HAS') + ' shape formula)'
                for m in _current_module_names
            ]
            return '\n'.join(current_modules_info)

        if formula is not None:
            try:
                shapes = formula(func, *args, **kwargs)
                rs = _assign_shape_info(rs, shapes)
            except:
                _logger.warning(
                    'Shape inference of %s failed though shape inference formula found: %s. Module calling stack:\n%s\n'
                    'This is either a bug in the shape inference formula, or something goes wrong earlier. '
                    'To workaround this, you can explicitly write a formula for its parent modules.\n%s',
                    func, formula, format_module_stack(), traceback.format_exc()
                )
        else:
            if not any(m.has_formula for m in _current_module_names):
                msg = 'and none of its parent modules have a shape inference formula. Shape inference is likely to fail. '
            else:
                msg = 'and a recent module that needs shape information has no shape inference formula. '
            _logger.warning(f'Shape information is not explicitly propagated when executing {func}, and {msg}'
                            'Module calling stack:\n' + format_module_stack())

        return rs

    @property
    def shape(self) -> tuple[IntProxy | int, ...]:
        """Alias of :meth:`size`."""
        return cast(Tuple[Union[IntProxy, int], ...], self.size())

    def size(self, dim: int | None = None) -> IntProxy | int | tuple[IntProxy | int, ...]:
        """Overrides the size method of Tensor as it's most important to us.

        It should behave similarly to the original size method,
        except the return value could be a tuple and each element is an :class:`IntProxy`.

        Since the return value of size is often used as integers in user code,
        we must return a tuple of :class:`IntProxy` instead of raw mutable expressions,
        so that they can be seamlessly used in user code.
        """
        # These branches can't be merged together because
        # `tensor.size()`` is not equivalent to `tensor.size(None)``.
        if dim is None:
            original_size = self.elem.size()
            if self.real_shape is None:
                # The real shape is unknown. We can't do anything.
                return original_size
            assert len(original_size) == len(self.real_shape)
            return tuple(IntProxy(s, e) for s, e in zip(original_size, self.real_shape))
        else:
            original_size = self.elem.size(dim)
            if self.real_shape is None:
                return original_size
            original_size = self.elem.size(dim)
            return IntProxy(original_size, self.real_shape[dim])

    def view(self, *shape: IntProxy | int) -> ShapeTensor:
        """:meth:`view` is very similar to ``Tensor.view()``
        except the input shape could be :class:`IntProxy` and the output shape is :class:`ShapeTensor`.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]  # type: ignore
        rv = self.elem.view(*shape)
        if self.real_shape is None:
            return ShapeTensor(rv)
        return self._view_impl(rv, *shape)

    def reshape(self, *shape: IntProxy | int) -> ShapeTensor:
        """Similar to ``Tensor.shape()`` except shapes can be mutables."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]  # type: ignore
        rv = self.elem.reshape(*shape)
        if self.real_shape is None:
            return ShapeTensor(rv)
        return self._view_impl(rv, *shape)

    def _view_impl(self, tensor: torch.Tensor, *shape: IntProxy | int) -> ShapeTensor:
        if any(s == -1 for s in shape):
            # the size -1 is inferred from other dimensions
            assert self.real_shape is not None
            all_size = self.real_shape.numel()
            shape_lst = list(shape)
            infer_index = shape_lst.index(-1)  # type: ignore
            assert infer_index != -1
            # Filter out -1, unwrap to get expression and multiply them together.
            shape_lst[infer_index] = all_size // functools.reduce(  # type: ignore
                lambda x, y: x * y,
                map(IntProxy.unwrap, filter(lambda x: x != -1, shape)),
                1,
            )
            shape = tuple(shape_lst)

        # Put in after "if" because `== -1` check doesn't work for mutables.
        return ShapeTensor(tensor, tuple(IntProxy.unwrap(s) for s in shape))


def submodule_input_output_shapes(
    model: nn.Module, *args: ShapeTensor,
    is_leaf: Callable[[nn.Module], bool] | None = None, **kwargs: ShapeTensor
) -> dict[str, tuple[MutableShape, MutableShape]]:
    """Get the dict of all the symbolic shapes of the inputs and outputs of all the submodules.

    Parameters
    ----------
    model
        The top-level module.
    *args
        Positional arguments to the model's forward.
    **kwargs
        Keyword arguments.
    is_leaf
        A function that takes a module and returns whether the module is a leaf module.
        The submodule of a leaf module will not appear in the result dict,
        which means we won't complain about their shapes even if missing (though there might be still warnings).
        By default, a module is a leaf module if it has no children or it's defined in torch.nn or its parameterized version.
        See :func:`~nni.nas.pytorch.utils.is_leaf_module` for details.
    """
    handles, shapes = _register_shape_inference_hooks(model, skip_toplevel=False, is_leaf=is_leaf)

    with torch.no_grad():
        model(*args, **kwargs)

    for handle in handles:
        handle.remove()

    return shapes


def shape_inference(module: nn.Module, *args: ShapeTensor,
                    is_leaf: Callable[[nn.Module], bool] | None = None, **kwargs: ShapeTensor) -> ShapeTensor | tuple[ShapeTensor, ...]:
    """
    Running the forward of module with :class:`ShapeTensor` as inputs,
    and return the output :class:`ShapeTensor`.

    Parameters
    ----------
    module
        The module to run.
    *args
        The inputs to the module.
    **kwargs
        The keyword arguments to the module.
    is_leaf
        A function that takes a module and returns whether the module is a leaf module.

    See Also
    --------
    submodule_input_output_shapes
    """
    handles, _ = _register_shape_inference_hooks(module, is_leaf=is_leaf)

    with torch.no_grad():
        outputs = module(*args, **kwargs)
    result = _module_shape_inference_impl(module, outputs, *args, **kwargs)

    for handle in handles:
        handle.remove()
    return result


def module_shape_inference_hook(module: nn.Module, input: Any, output: Any,
                                is_leaf: Callable[[nn.Module], bool] | None = None,
                                save_io_callback: Callable[..., None] | None = None) -> Any:
    """
    A hook that runs the shape inference on the module.
    The hook can't process keyword arguments due to the limitation of PyTorch.
    But I think it will be resolved `soon <https://github.com/pytorch/pytorch/pull/89389>`__.
    """
    result = _module_shape_inference_impl(module, output, *input, is_leaf=is_leaf)

    if save_io_callback is not None:
        save_io_callback(
            extract_shape_info(input),
            extract_shape_info(result)
        )

    return result


def _register_shape_inference_hooks(
    module: nn.Module, skip_toplevel: bool = True, is_leaf: Callable[[nn.Module], bool] | None = None
) -> tuple[list[RemovableHandle], dict[str, tuple[MutableShape, MutableShape]]]:
    """This method registers every submodule of ``module`` with a hook,
    so that when running forward, the results carry the shape information.

    This method only needs to be called once for a module, though it can be called multiple times.

    Parameters
    ----------
    module
        The top-level module.
    skip_toplevel
        Whether to wrap the forward of the top-level module.
    is_leaf
        When ``is_leaf`` returns true, its inner submodules will not be wrapped with a hook.

    Returns
    -------
    Handles to remove the hooks,
    and a dict that will show the input/output shapes of each submodule after running forward.
    """
    # Handles are used to remove the hooks when exiting.
    handles: list[RemovableHandle] = []
    # Input and output shapes shapes are used to save the shapes of inputs for each submodule.
    # Mapping is from submodule name to input shapes.
    shapes: dict[str, tuple[MutableShape, MutableShape]] = {}

    if is_leaf is None:
        is_leaf = is_leaf_module

    def save_io(name: str, input: Any, output: Any) -> None:
        shapes[name] = (input, output)

    def _iter(name: str, mod: nn.Module) -> None:
        # For self

        # Skip the top-level module so that we can at least process keyword arguments for the toppest.
        if not skip_toplevel or name != '':
            # Save input/output shapes for the specific submodule.
            save_io_callback = functools.partial(save_io, name)
            # ... and create the hook.
            hook = functools.partial(
                module_shape_inference_hook,
                is_leaf=is_leaf,
                save_io_callback=save_io_callback
            )
            # We don't check whether the hook has already been registered here because it's not a big deal,
            # and it will be hard to check whether the hook should be unregistered when exiting.
            handles.append(mod.register_forward_hook(hook))

        handles.append(mod.register_forward_pre_hook(functools.partial(_current_module_pre_hook, name)))
        handles.append(mod.register_forward_hook(functools.partial(_current_module_post_hook, name)))

        # For children
        if is_leaf is None or not is_leaf(mod):
            for child_name, child in mod.named_children():
                _iter(name + '.' + child_name if name else child_name, child)

    _iter('', module)

    # shapes contain nothing at this point. It will be filled when forward is called.
    return handles, shapes


def _module_shape_inference_impl(
    module: nn.Module, outputs: Any, *input_args: ShapeTensor,
    is_leaf: Callable[[nn.Module], bool] | None = None, **input_kwargs: ShapeTensor
) -> ShapeTensor | tuple[ShapeTensor, ...]:

    from torch.utils._pytree import tree_map

    from .shape_formula import find_shape_inference_formula

    formula = find_shape_inference_formula(module)

    if formula is None:
        # No formula, but the shapes could've been already inferred.
        def _ensure_shape(tensor: ShapeTensor) -> None:
            if isinstance(tensor, ShapeTensor) and tensor.real_shape is None:
                # TODO: more informative error message
                module_repr = repr(module)
                if len(module_repr) > 100:
                    module_repr = module_repr[:100] + '...'
                raise RuntimeError(
                    f'Shape inference failed because no shape inference formula is found for {module_repr} '
                    f'of type {type(module).__name__}. '
                    'Meanwhile the nested modules and functions inside failed to propagate the shape information. '
                    'Please provide a `_shape_forward` member function or register a formula using `register_shape_inference_formula`.'
                )

        # Check every ShapeTensor in outputs
        tree_map(_ensure_shape, outputs)
        return outputs

    # If we find a formula, we must run the formula to find out the shape,
    # because the shapes attached to the tensor could not be desired.

    # Some formulas support an optional argument `is_leaf`.
    formula_kwargs = {}
    if argument_in_spec(formula, 'is_leaf'):
        formula_kwargs['is_leaf'] = is_leaf

    output_shape = formula(module, *input_args, **formula_kwargs, **input_kwargs)
    outputs = _assign_shape_info(outputs, output_shape)
    return outputs


# Utility methods to update current module stack. For verbose info when something goes wrong.

def _current_module_pre_hook(name: str, module: nn.Module, *args, **kwargs) -> None:
    from .shape_formula import find_shape_inference_formula
    # Sometimes the same module has multiple shape inference hooks.
    if _current_module_names and _current_module_names[-1].id == id(module):
        return
    _current_module_names.append(_ModuleName(
        id(module), name, type(module).__module__ + '.' + type(module).__name__,
        find_shape_inference_formula(module) is not None
    ))


def _current_module_post_hook(name: str, module: nn.Module, *args, **kwargs) -> None:
    # Must match before popping.
    # Otherwise it might pop the wrong module in case there are multiple hooks for one module.
    if _current_module_names and _current_module_names[-1].id == id(module) and _current_module_names[-1].name == name:
        _current_module_names.pop()
