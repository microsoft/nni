# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = ['register_shape_inference_formula', 'find_shape_inference_formula']

import logging
import warnings
from typing import Callable, Type, Tuple, Any, cast

import torch
from torch import nn

import nni.nas.nn.pytorch as nas_nn
from nni.mutable import MutableExpression
from .shape import Formula, ShapeTensor, MutableShape, extract_shape_info, switch_case_shape_info, shape_inference
from ._attrs import _getattr, tuple_2_t

_logger = logging.getLogger(__name__)

# Raise warnings for aten operator formulas.
# _aten_warning = 'Aten operator detected. The shape inference formula for this operator is limited. ' \
#     'Please consider using modules from torch.nn instead. Suspected use of functions: {suspect}'


def register_shape_inference_formula(class_or_func: Any, formula: Formula) -> None:
    """
    Register a shape inference formula for a module.

    Parameters
    ----------
    class_or_func
        The module or function to register the formula for.
        The class here needs to be a class, not an instantiated module.
    formula
        A function that takes in a module and its inputs, and returns the output shape.
        To be specific, its input will be the module or function itself,
        plus ``*args`` and ``**kwargs`` of the module or function.
        Tensors will be replaced with :class:`ShapeTensor` objects.
        The output should be the same format as its normal forward output,
        but every tensors should be replaced with a :class:`MutableShape` object.

    Examples
    --------
    Here is an example of a formula for FC::

        def linear_formula(module: nn.Linear, input: ShapeTensor) -> MutableShape:
            return MutableShape(*tuple(input.real_shape)[:-1], module.out_features)

    It can be registered with::

        register_shape_inference_formula(nn.Linear, linear_formula)
    """
    if class_or_func in _shape_inference_formulas:
        _logger.warning(f'Overwriting shape inference formula for {class_or_func}')
    _shape_inference_formulas[class_or_func] = formula


def find_shape_inference_formula(module_or_func: Any) -> Formula | None:
    """
    Find the shape inference formula for a module or function.

    It searches two places in order:

    1. The module's ``_shape_forward`` attribute.
       The function should follow the signature defined in :func:`register_shape_inference_formula`.
    2. The global registry. Register with :func:`register_shape_inference_formula`.

    Parameters
    ----------
    module_or_func
        The module or function to find the formula for.
        The module here needs to be an instantiated module, not a class.
    """

    if isinstance(module_or_func, nn.Module):
        formula = None
        # Shape inference formulas are fetched in the following order:
        # 1. Check `_shape_forward` of the module.
        if hasattr(module_or_func.__class__, '_shape_forward'):
            formula: Any = module_or_func.__class__._shape_forward  # type: ignore
        # 2. Check global registered formulas.
        elif type(module_or_func) in _shape_inference_formulas:
            formula = _shape_inference_formulas[type(module_or_func)]
        return formula
    else:
        return _shape_inference_formulas.get(module_or_func)


def _safe_register_aten_formula(name: str, formula: Formula) -> None:
    """Register a shape inference formula for an aten operator.

    Some aten operators are internal and not trusted to be stable.
    This function will raise a warning if the operator is not found.
    """
    # For legacy PyTroch
    suffixes = ['.default', '.Tensor', '.dim', '.int']
    if any(name.endswith(suffix) for suffix in suffixes):
        _safe_register_aten_formula(name.rsplit('.', 1)[0], formula)

    names = name.split('.')
    object = torch.ops.aten
    for name in names:
        try:
            if not hasattr(object, name):
                warnings.warn(f'Cannot find a {name} in torch.ops.aten because {object} has no attribute {name}. '
                              'Skip registering the shape inference formula.')
                return
        except RuntimeError as e:
            # Some pytorch version will raise RuntimeError when using hasattr
            warnings.warn(f'Fail to register shape inference formula for aten operator {name} because: {e}')
            return
        object = getattr(object, name)
    register_shape_inference_formula(object, formula)


# General formulas for operators that do not change the shape.

def ensure_shape(input: ShapeTensor) -> MutableShape:
    if input.real_shape is not None:
        return input.real_shape
    raise ValueError(f'Shape of input is not known: f{input}')


def keep_shape_formula(any_callable: Any, *args, **kwargs) -> MutableShape:
    if len(args) == 1:
        return extract_shape_info(args[0])
    return extract_shape_info(args)


def keep_first_shape_formula(any_callable: Any, *args, **kwargs) -> MutableShape:
    return extract_shape_info(args[0])


# Formulas for modules.

def linear_formula(module: nn.Linear | nas_nn.MutableLinear, input: ShapeTensor) -> MutableShape:
    assert input.real_shape is not None
    out_features = _getattr(module, 'out_features')
    return MutableShape(*tuple(input.real_shape)[:-1], out_features)


def conv2d_formula(module: nn.Conv2d | nas_nn.MutableConv2d, input: ShapeTensor) -> MutableShape:
    shape = list(input.real_shape)  # type: ignore

    out_channels = _getattr(module, 'out_channels')
    padding, dilation, kernel_size, stride = map(
        lambda name: _getattr(module, name, expected_type=tuple_2_t),
        ['padding', 'dilation', 'kernel_size', 'stride']
    )

    # C_out
    shape[-3] = out_channels
    # H_out and W_out
    shape[-2] = (shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    shape[-1] = (shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return MutableShape(*shape)


def maxpool2d_formula(module: nn.MaxPool2d | nas_nn.MutableMaxPool2d, input: ShapeTensor) -> MutableShape:
    shape = list(input.real_shape)  # type: ignore

    padding, dilation, kernel_size, stride = map(
        lambda name: _getattr(module, name, expected_type=tuple_2_t),
        ['padding', 'dilation', 'kernel_size', 'stride']
    )

    # H_out and W_out
    shape[-2] = (shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    shape[-1] = (shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return MutableShape(*shape)

def avgpool2d_formula(module: nn.AvgPool2d , input: ShapeTensor) -> MutableShape:
    shape = list(input.real_shape)  # type: ignore 

    padding, kernel_size, stride = map(
        lambda name: _getattr(module, name, expected_type=tuple_2_t),
        ['padding', 'kernel_size', 'stride']
    ) 
  
    # H_out and W_out
    shape[-2] = (shape[-2] + 2 * padding[0] -  (kernel_size[0])) // stride[0] + 1
    shape[-1] = (shape[-1] + 2 * padding[1] -  (kernel_size[1])) // stride[1] + 1
    return MutableShape(*shape)

def multihead_attention_formula(module: nn.MultiheadAttention | nas_nn.MutableMultiheadAttention,
                                query: ShapeTensor, key: ShapeTensor, *args: Any, **kwargs) -> tuple[MutableShape, MutableShape | None]:
    shape = list(query.real_shape)  # type: ignore
    attn_shape = MutableShape(*shape[:-1], _getattr(module, 'embed_dim'))
    key_shape = ensure_shape(key)

    weights_shape = None
    if kwargs.get('need_weights', True):
        batch_first = module.batch_first
        if module.batch_first is not _getattr(module, 'batch_first'):
            _logger.warning('The batch_first attribute of the module is different from the batch_first '
                            'attribute of the formula. The shape inference result may be incorrect. '
                            'Assuming batch_first to be %s.', batch_first)

        if len(shape) == 2:
            # unbatched
            N, L = None, shape[0]
            S = key_shape[0]
        elif batch_first:
            N, L = shape[0], shape[1]
            S = key_shape[1]
        else:
            L, N = shape[0], shape[1]
            S = key_shape[0]

        if kwargs.get('average_attn_weights', True):
            if N is None:
                weights_shape = MutableShape(L, S)
            else:
                weights_shape = MutableShape(N, L, S)
        else:
            num_heads = _getattr(module, 'num_heads')
            if N is None:
                weights_shape = MutableShape(num_heads, L, S)
            else:
                weights_shape = MutableShape(N, num_heads, L, S)

    return attn_shape, weights_shape


def adaptive_avg_pool2d_formula(module: nn.AdaptiveAvgPool2d | nas_nn.MutableAdaptiveAvgPool2d,
                                input: ShapeTensor) -> MutableShape:
    shape = list(input.real_shape)  # type: ignore
    output_size = _getattr(module, 'output_size', expected_type=tuple_2_t)
    shape[-2] = output_size[0]
    shape[-1] = output_size[1]
    return MutableShape(*shape)


# Formulas for nni mutable modules.

def layer_choice_formula(module: nas_nn.LayerChoice, *args: ShapeTensor,
                         is_leaf: Callable[[nn.Module], bool] | None = None, **kwargs: ShapeTensor) -> MutableShape:
    expressions = {}
    for val in module.choice.values:
        expressions[val] = extract_shape_info(shape_inference(module[val], *args, is_leaf=is_leaf, **kwargs))
    return switch_case_shape_info(module.choice, expressions)


def input_choice_formula(module: nas_nn.InputChoice, input_tensors: list[ShapeTensor]) -> MutableShape:
    if module.n_chosen != 1:
        raise ValueError(f'Input choice with multiple choices (e.g., n_chosen = {module.n_chosen}) is not supported yet.')
    assert len(input_tensors) > 0
    for tensor in input_tensors:
        if tensor.real_shape != input_tensors[0].real_shape:
            _logger.warning('Expected all input tensors to a input choice to have the same input shape, but found ',
                            '%s vs. %s', tensor.real_shape, input_tensors[0].real_shape)
    return extract_shape_info(input_tensors[0])


def repeat_formula(module: nas_nn.Repeat, input: ShapeTensor, is_leaf: Callable[[nn.Module], bool] | None = None) -> Tuple[MutableShape, ...]:
    if isinstance(module.depth_choice, int):
        for sub in module:
            input = cast(ShapeTensor, shape_inference(sub, input, is_leaf=is_leaf))
        return extract_shape_info(input)
    else:
        possible_depths = sorted(set(module.depth_choice.grid()))
        expressions = {}
        if 0 in possible_depths:
            expressions[0] = extract_shape_info(input)
        for depth, sub in enumerate(module.blocks, start=1):
            input = cast(ShapeTensor, shape_inference(sub, input, is_leaf=is_leaf))
            if depth in possible_depths:
                expressions[depth] = extract_shape_info(input)
        return switch_case_shape_info(module.depth_choice, expressions)


# Follows definitions from:
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml

def _canonicalize_dims(dims: list[int], n_dims: int, fn: Any) -> list[int]:
    if any(not isinstance(d, int) for d in dims):
        raise RuntimeError(f'Cannot infer the shape of {fn} because the input dims are not all integers: {dims}. ')
    return [d if d >= 0 else d + n_dims for d in dims]


def aten_reshape_alias_formula(fn: Any, input: ShapeTensor, size: list[int], stride: list[int] | None = None) -> MutableShape:
    input_shape = ensure_shape(input)
    if input_shape.is_mutable():
        raise RuntimeError(f'Cannot infer the shape of {fn} because the input shape is not determined: {input_shape}, '
                           f'but output shape is fixed: {size}. '
                           'This happens when functions like `torch.flatten` is used on a mutable-shape input. '
                           'Try to use `.view()` instead.')

    return MutableShape(*size)


def aten_mean_dim(fn: Any, input: ShapeTensor, dim: list[int], keepdim: bool = False, **kwargs) -> MutableShape:
    input_shape = ensure_shape(input)
    dim = _canonicalize_dims(dim, len(input_shape), fn)
    if keepdim:
        shape = [1 if i in dim else s for i, s in enumerate(input_shape)]
        return MutableShape(*shape)
    else:
        return MutableShape(*[s for i, s in enumerate(input_shape) if i not in dim])


def aten_shape_broadcast(fn: Any, x: ShapeTensor, y: ShapeTensor, **kwargs) -> MutableShape:
    # Rules are:
    # 1. If the number of dimensions of x and y are not equal,
    #    prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
    # 2. Merge the shape of x and y. Avoid 1.

    if y.real_shape is None:
        return cast(MutableShape, x.real_shape)  # Assume x has shape.
    if x.real_shape is None:
        return cast(MutableShape, y.real_shape)

    x_shape = list(x.real_shape)
    y_shape = list(y.real_shape)

    if len(x_shape) > len(y_shape):
        y_shape = [1] * (len(x_shape) - len(y_shape)) + y_shape  # type: ignore
    elif len(x_shape) < len(y_shape):
        x_shape = [1] * (len(y_shape) - len(x_shape)) + x_shape  # type: ignore
    assert len(x_shape) == len(y_shape)

    # Use "is" here to avoid generating mutable expressions.
    ONE = 1
    return MutableShape(*[y if x is ONE else x for x, y in zip(x_shape, y_shape)])


def aten_permute_formula(fn: Any, input: ShapeTensor, dims: list[int], **kwargs) -> MutableShape:
    input_shape = ensure_shape(input)
    dims = _canonicalize_dims(dims, len(input_shape), fn)
    return MutableShape(*[input_shape[d] for d in dims])


def aten_slice_formula(fn: Any, input: ShapeTensor, dim: int, start: int, end: int, step: int = 1, **kwargs) -> MutableShape:
    input_shape = ensure_shape(input)
    dim = _canonicalize_dims([dim], len(input_shape), fn)[0]
    start, end = _canonicalize_dims([start, end], input_shape[dim], fn)  # type: ignore
    assert start >= 0 and end >= start and step > 0, f'Unsupported slice range: {start} {end} {step}'
    end = MutableExpression.min(end, input_shape[dim])
    return MutableShape(*[s if i != dim else (end - start) // step for i, s in enumerate(input_shape)])


def aten_select_formula(fn: Any, input: ShapeTensor, dim: int, index: int, **kwargs) -> MutableShape:
    input_shape = ensure_shape(input)
    dim = _canonicalize_dims([dim], len(input_shape), fn)[0]
    return MutableShape(*[s for i, s in enumerate(input_shape) if i != dim])


def aten_cat_formula(fn: Any, input: list[ShapeTensor], dim: int = 0, **kwargs) -> MutableShape:
    first_input_shape = cast(MutableShape, input[0].real_shape)
    dim = _canonicalize_dims([dim], len(first_input_shape), fn)[0]
    result = list(first_input_shape)
    result[dim] = sum(t.real_shape[dim] for t in input)  # type: ignore
    return MutableShape(*result)


_shape_inference_formulas: dict[Type[nn.Module], Formula] = {
    nn.Linear: linear_formula,
    nn.Conv2d: conv2d_formula,
    nn.MaxPool2d: maxpool2d_formula,
    nn.AvgPool2d: avgpool2d_formula,
    nn.BatchNorm2d: keep_shape_formula,
    nn.LayerNorm: keep_shape_formula,
    nn.MultiheadAttention: multihead_attention_formula,
    nn.AdaptiveAvgPool2d: adaptive_avg_pool2d_formula,
    nn.Dropout: keep_shape_formula,

    nas_nn.MutableLinear: linear_formula,
    nas_nn.MutableConv2d: conv2d_formula,
    nas_nn.MutableMaxPool2d: maxpool2d_formula,
    nas_nn.MutableBatchNorm2d: keep_shape_formula,
    nas_nn.MutableLayerNorm: keep_shape_formula,
    nas_nn.MutableMultiheadAttention: multihead_attention_formula,
    nas_nn.MutableAdaptiveAvgPool2d: adaptive_avg_pool2d_formula,
    nas_nn.MutableDropout: keep_shape_formula,

    # Special NAS modules
    nas_nn.LayerChoice: layer_choice_formula,
    nas_nn.InputChoice: input_choice_formula,
    nas_nn.Repeat: repeat_formula,
}

_safe_register_aten_formula('relu.default', keep_shape_formula)
_safe_register_aten_formula('gelu.default', keep_shape_formula)
_safe_register_aten_formula('hardswish.default', keep_shape_formula)
_safe_register_aten_formula('hardsigmoid.default', keep_shape_formula)
_safe_register_aten_formula('relu_.default', keep_shape_formula)
_safe_register_aten_formula('hardswish_.default', keep_shape_formula)
_safe_register_aten_formula('hardsigmoid_.default', keep_shape_formula)
_safe_register_aten_formula('hardtanh_.default', keep_first_shape_formula)
_safe_register_aten_formula('permute.default', aten_permute_formula)
_safe_register_aten_formula('select.int', aten_select_formula)
_safe_register_aten_formula('cat.default', aten_cat_formula)
_safe_register_aten_formula('mean.dim', aten_mean_dim)
_safe_register_aten_formula('_log_softmax.default', keep_first_shape_formula)
_safe_register_aten_formula('_reshape_alias.default', aten_reshape_alias_formula)
_safe_register_aten_formula('view.default', aten_reshape_alias_formula)
_safe_register_aten_formula('add.Tensor', aten_shape_broadcast)
_safe_register_aten_formula('mul.Tensor', aten_shape_broadcast)
_safe_register_aten_formula('slice.Tensor', aten_slice_formula)
