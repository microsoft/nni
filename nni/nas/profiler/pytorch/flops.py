# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

__all__ = [
    'FlopsParamsCounterConfig', 'FlopsParamsProfiler', 'FlopsProfiler', 'NumParamsProfiler',
    'count_flops_params', 'register_flops_formula'
]

import logging
from dataclasses import dataclass, fields
from functools import partial, reduce
from typing import Any, NamedTuple, Callable, cast

from torch import nn

import nni.nas.nn.pytorch as nas_nn
from nni.mutable import Sample, Mutable, MutableExpression
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.profiler import Profiler, ExpressionProfiler

from .utils import MutableShape, ShapeTensor, submodule_input_output_shapes, standardize_arguments, argument_in_spec, concat_name
from .utils._attrs import tuple_n_t, _getattr

_unlisted_types = set()

_logger = logging.getLogger(__name__)


@dataclass
class FlopsParamsCounterConfig:
    """
    Configuration for counting FLOPs.

    Attributes
    ----------
    count_bias
        Whether to count bias into FLOPs.
    count_normalization
        Whether to count normalization (e.g., Batch normalization) into FLOPs **and parameters**.
    count_activation
        Whether to count activation (e.g., ReLU) into FLOPs.
    """

    count_bias: bool = True
    count_normalization: bool = True
    count_activation: bool = True


def count_flops_params(name: str, module: nn.Module, shapes: dict[str, tuple[MutableShape, MutableShape]],
                       config: FlopsParamsCounterConfig) -> FlopsResult:
    """
    Count FLOPs of a module.

    Firstly check whether the type of module is in FLOPs formula registry.
    If not, traverse its children and sum up the FLOPs of each child.

    Parameters
    ----------
    name
        Name of the module.
    module
        The module to count FLOPs.
    shapes
        Input and output shapes of all the modules.
        Should at least contain ``name``.

    Returns
    -------
    flops
        The FLOPs of the module.
    """
    formula = find_flops_formula(module)

    if formula is not None:
        if name not in shapes:
            raise KeyError(f'Cannot find shape of {name} in shapes. This could be caused by a shape inference that is too coarse-grained, '
                           f'or the module did not appear in the forward at all. Existing modules are:\n  {list(shapes.keys())}')

        # We provide arguments to the formula when they need them.
        formula_kwargs = {}
        if argument_in_spec(formula, 'name'):
            formula_kwargs['name'] = name
        if argument_in_spec(formula, 'shapes'):
            formula_kwargs['shapes'] = shapes
        if argument_in_spec(formula, 'config'):
            formula_kwargs['config'] = config
        for field in fields(config):
            if argument_in_spec(formula, field.name):
                formula_kwargs[field.name] = getattr(config, field.name)

        rv = formula(module, *shapes[name], **formula_kwargs)
        _logger.debug('FLOPs of %s counts to: %s', name, rv)
        return rv

    else:
        children_flops = [
            count_flops_params(concat_name(name, n), child, shapes, config)
            for n, child in module.named_children()
        ]
        if children_flops:
            rv = sum(children_flops)
            _logger.debug('FLOPs of %s sums up to be: %r', name, rv)
            return cast(FlopsResult, rv)
        else:
            # Leaf module. No children.
            if type(module) not in _unlisted_types:
                _logger.warning('Parameters and FLOPs of %s is not counted because it has no children (name: %s).',
                                type(module).__name__, name)
                _unlisted_types.add(type(module))
            return FlopsResult(0., 0.)


class FlopsParamsProfiler(Profiler):
    _parameters_doc = """

    Parameters
    ----------
    model_space
        The model space to profile.
    args
        Dummy inputs to the model to count flops.
        Similar to `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`__,
        the input can be a tensor or a tuple of tensors, or a tuple of arguments ends with a dictionary of keyword arguments.
    **kwargs
        Additional configurations. See :class:`FlopsParamsCounterConfig` for supported arguments.
"""

    __doc__ = """
    The profiler to count flops and parameters of a model.

    It first runs shape inference on the model to get the input/output shapes of all the submodules.
    Then it traverse the submodules and use registered formulas to count the FLOPs and parameters as an expression.
    The results are stored in a :class:`FlopsResult` object.
    When a sample is provided, the expressions are frozen and the results are computed.

    Notes
    -----
    Customized FLOPs formula can be registered by using :func:`register_flops_formula`.
    It takes three mandatory arguments: the module itself, input shapes as a tuple of :class:`MutableShape` objects,
    and output shapes as a tuple of :class:`MutableShape` objects.
    It also takes some additional keyword arguments:

    - ``name``: the name of the module in the PyTorch module hierarchy.
    - ``shapes``: a dictionary of all the input and output shapes of all the modules.
    - ``config``: the configuration object of :class:`FlopsParamsProfiler`.

    If fields in :class:`FlopsParamsCounterConfig` are used in the formula, they will also be passed as keyword arguments.

    It then returns a :class:`FlopsResult` object that contains the FLOPs and parameters of the module.

    For example, to count the FLOPs of a unbiased linear layer, we can register the following formula::

        def linear_flops(module, input_shape, output_shape, *, name, shapes, config):
            x, y = input_shape[0], output_shape[0]  # unpack the tuple
            return FlopsResult(
                flops=x[1:].numel() * module.out_features,  # forget the batch size
                params=module.in_features * module.out_features
            )

        register_flops_formula(nn.Linear, linear_flops)

    """ + _parameters_doc

    def __init__(self, model_space: ModelSpace, args: Any, **kwargs):
        self.config = FlopsParamsCounterConfig(**kwargs)

        args, kwargs = standardize_arguments(args, lambda t: ShapeTensor(t, True))

        shapes = submodule_input_output_shapes(model_space, *args, **kwargs)

        self.flops_result = count_flops_params('', model_space, shapes, self.config)

    def profile(self, sample: Sample) -> FlopsResult:
        return self.flops_result.freeze(sample)


class FlopsProfiler(ExpressionProfiler):
    __doc__ = """
    The FLOPs part of :class:`FlopsParamsProfiler`.

    Batch size is not considered (actually ignored on purpose) in flops profiling.
    """ + FlopsParamsProfiler._parameters_doc

    def __init__(self, model_space: ModelSpace, args: Any, **kwargs: Any):
        self.profiler = FlopsParamsProfiler(model_space, args, **kwargs)
        self.expression = self.profiler.flops_result.flops


class NumParamsProfiler(ExpressionProfiler):
    __doc__ = """
    The parameters part of :class:`FlopsParamsProfiler`.
    """ + FlopsParamsProfiler._parameters_doc

    def __init__(self, model_space: ModelSpace, args: Any, **kwargs: Any):
        self.profiler = FlopsParamsProfiler(model_space, args, **kwargs)
        self.expression = self.profiler.flops_result.params


class FlopsResult(NamedTuple):
    """The result of flops profiling. Tuple of (flops, params).

    It has a :meth:`freeze` that mimics the behavior of :class:`MutableExpression`,
    and returns a frozen :class:`FlopsResult` object when a sample is provided.
    """

    flops: float | MutableExpression[float]
    params: float | MutableExpression[float]

    def __add__(self, other):
        if isinstance(other, FlopsResult):
            return FlopsResult(self.flops + other.flops, self.params + other.params)
        elif isinstance(other, (int, float)):
            return FlopsResult(self.flops + other, self.params)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return FlopsResult(other + self.flops, self.params)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(other)}' and '{type(self)}'")

    def __eq__(self, other):
        if isinstance(other, FlopsResult):
            return self.flops == other.flops and self.params == other.params
        else:
            return False

    def freeze(self, sample: Sample) -> FlopsResult:
        flops = self.flops.freeze(sample) if isinstance(self.flops, Mutable) else self.flops
        params = self.params.freeze(sample) if isinstance(self.params, Mutable) else self.params
        return FlopsResult(flops, params)


def _count_element_size(module: Any, input: tuple[MutableShape, ], output: tuple[MutableShape, ]) -> FlopsResult:
    x = input[0]
    total_ops = x[1:].numel()
    return FlopsResult(total_ops, 0)


def _count_activation(module: Any, input: tuple[MutableShape, ], output: tuple[MutableShape, ],
                      count_activation: bool = True) -> FlopsResult:
    if not count_activation:
        return FlopsResult(0., 0.)
    return _count_element_size(module, input, output)


def _count_convNd(
    module: nn.Conv1d | nn.Conv2d | nn.Conv3d | nas_nn.MutableConv1d | nas_nn.MutableConv2d | nas_nn.MutableConv3d,
    input: tuple[MutableShape, ], output: MutableShape, N: int, count_bias: bool = True
) -> FlopsResult:
    cin = _getattr(module, 'in_channels')
    cout = _getattr(module, 'out_channels')
    kernel_ops = reduce(lambda a, b: a * b, _getattr(module, 'kernel_size', expected_type=tuple_n_t[N]))
    bias = _getattr(module, 'bias', expected_type=bool)
    groups = _getattr(module, 'groups')
    resolution_out = output[-N:].numel()

    total_ops = cout * resolution_out * (kernel_ops * cin // groups + (bias and count_bias))  # cout x oW x oH
    total_params = cout * (cin // groups * kernel_ops + bias)  # always count bias

    return FlopsResult(total_ops, total_params)


def _count_linear(
    module: nn.Linear | nas_nn.Linear,
    input: tuple[MutableShape, ], output: MutableShape,
    count_bias: bool = True
) -> FlopsResult:
    in_features = _getattr(module, 'in_features')
    out_features = _getattr(module, 'out_features')
    bias = _getattr(module, 'bias', expected_type=bool)
    total_ops = out_features * (in_features + (bias and count_bias))
    if len(input[0]) > 2:  # more than batch and hidden_dim
        total_ops = total_ops * input[0][1:-1].numel()
    total_params = out_features * (in_features + bias)

    return FlopsResult(total_ops, total_params)


def _count_bn(module: nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d |
              nas_nn.MutableBatchNorm1d | nas_nn.MutableBatchNorm2d | nas_nn.MutableBatchNorm3d,
              input: tuple[MutableShape, ], output: MutableShape,
              count_normalization: bool = True) -> FlopsResult:
    if not count_normalization:
        return FlopsResult(0., 0.)
    affine = _getattr(module, 'affine', expected_type=bool)
    num_features = _getattr(module, 'num_features')

    x = input[0]
    total_ops = 2 * (1 + affine) * x[1:].numel()
    total_params = affine * 2 * num_features
    return FlopsResult(total_ops, total_params)


def _count_mhattn(module: nn.MultiheadAttention | nas_nn.MultiheadAttention,
                  input: tuple[MutableShape, MutableShape, MutableShape], output: MutableShape,
                  count_bias: bool = True) -> FlopsResult:
    # https://github.com/sovrasov/flops-counter.pytorch/blob/0d9518e2ca4cae9b2a0f43dd8c3297c30e33a565/ptflops/pytorch_ops.py#L159

    q, k, v = input[:3]  # there could be more than 3 arguments.

    qdim, kdim, vdim = q[-1], k[-1], v[-1]
    if not module.batch_first or len(q) == 2:
        # No batch dimension or batch dimension is the second dimension
        L, S = q[0], v[0]
    else:
        L, S = q[1], v[1]

    num_heads = _getattr(module, 'num_heads')
    embed_dim = _getattr(module, 'embed_dim')
    try:
        bias = _getattr(module, 'bias', expected_type=bool)  # Mutable version
    except AttributeError:
        bias = _getattr(module, 'in_proj_bias', expected_type=bool)  # nn version
    head_dim = embed_dim // num_heads

    params = (                                       # always count bias
        (qdim + kdim + vdim) * (embed_dim + bias) +  # in_proj
        embed_dim * (embed_dim + bias)               # out_proj
    )

    flops = (
        L * qdim +                                   # Q scaling
        L * embed_dim * (qdim + (bias and count_bias)) +  # in_proj Q
        S * embed_dim * (kdim + (bias and count_bias)) +  # in_proj K
        S * embed_dim * (vdim + (bias and count_bias)) +  # in_proj V
        num_heads * (                                # attention heads: heads * (L * S * dim * 2 + 1)
            L * S * head_dim +                       # QK^T
            L * S +                                  # softmax
            L * S * head_dim                         # AV
        ) +
        L * embed_dim * (embed_dim + (bias and count_bias))  # out_proj
    )

    return FlopsResult(flops, params)


def _count_layerchoice(module: nas_nn.LayerChoice, input: tuple[MutableShape, ], output: MutableShape,
                       name: str, shapes: dict[str, tuple[MutableShape, MutableShape]],
                       config: FlopsParamsCounterConfig) -> FlopsResult:
    sub_results: dict[int | str, FlopsResult] = {}
    for chosen_val in module.choice.values:
        sub_results[chosen_val] = count_flops_params(
            concat_name(name, str(chosen_val)),
            module[chosen_val],
            shapes,
            config
        )
    return FlopsResult(
        MutableExpression.switch_case(module.choice, {chosen_val: res.flops for chosen_val, res in sub_results.items()}),
        MutableExpression.switch_case(module.choice, {chosen_val: res.params for chosen_val, res in sub_results.items()})
    )


def _count_repeat(module: nas_nn.Repeat, input: tuple[MutableShape, ], output: MutableShape,
                  name: str, shapes: dict[str, tuple[MutableShape, MutableShape]],
                  config: FlopsParamsCounterConfig) -> FlopsResult:
    if isinstance(module.depth_choice, int):
        assert module.depth_choice > 0
        return cast(FlopsResult, sum(
            count_flops_params(
                concat_name(name, f'blocks.{i}'),
                module.blocks[i],
                shapes,
                config
            ) for i in range(module.depth_choice)
        ))
    else:
        flops_results: list[MutableExpression] = []
        params_results: list[MutableExpression] = []
        for depth, sub in enumerate(module.blocks, start=1):
            sub_result = count_flops_params(
                concat_name(name, f'blocks.{depth - 1}'),
                sub, shapes, config
            )
            # Only activated when the chosen depth is greater than "depth".
            flops_results.append((module.depth_choice >= depth) * sub_result.flops)
            params_results.append((module.depth_choice >= depth) * sub_result.params)
        return FlopsResult(sum(flops_results), sum(params_results))


def register_flops_formula(
    module_type: Any,
    formula: Callable[..., FlopsResult]
) -> None:
    """
    Register a FLOPs counting formula for a module.

    Parameters
    ----------
    module_type
        The module type to register the formula for.
        The class here needs to be a class, not an instantiated module.
    formula
        A function that takes in a module and its inputs, and returns :class:`FlopsResult`.
        Check :class:`FlopsParamsProfiler` for more details.
    """
    if module_type in _flops_formula:
        _logger.warning('Formula for %s is already registered. It will be overwritten.', module_type)
    _flops_formula[module_type] = formula


def find_flops_formula(module: nn.Module) -> Callable[..., FlopsResult] | None:
    """
    Find the FLOPs counting formula for a module. In the following order:

    1. If the module has a ``_count_flops`` method, use it.
    2. If the module type is in the registry, use the registered formula.

    Parameters
    ----------
    module
        The module to find the formula for.

    Returns
    -------
    formula
        The FLOPs counting formula.
    """
    if hasattr(module.__class__, '_count_flops'):
        return module.__class__._count_flops  # type: ignore
    elif type(module) in _flops_formula:
        return _flops_formula[type(module)]  # type: ignore
    return None


_flops_formula = {
    nn.ReLU: _count_activation,
    nn.ReLU6: _count_activation,
    nn.ELU: _count_activation,
    nn.Sigmoid: _count_activation,
    nn.Tanh: _count_activation,
    nn.Hardtanh: _count_activation,
    nn.LeakyReLU: _count_activation,

    nn.Conv1d: partial(_count_convNd, N=1),
    nn.Conv2d: partial(_count_convNd, N=2),
    nn.Conv3d: partial(_count_convNd, N=3),
    nn.Linear: _count_linear,
    nn.BatchNorm1d: _count_bn,
    nn.BatchNorm2d: _count_bn,
    nn.BatchNorm3d: _count_bn,
    nn.MultiheadAttention: _count_mhattn,

    nas_nn.MutableConv1d: partial(_count_convNd, N=1),
    nas_nn.MutableConv2d: partial(_count_convNd, N=2),
    nas_nn.MutableConv3d: partial(_count_convNd, N=3),
    nas_nn.MutableLinear: _count_linear,
    nas_nn.MutableBatchNorm1d: _count_bn,
    nas_nn.MutableBatchNorm2d: _count_bn,
    nas_nn.MutableBatchNorm3d: _count_bn,
    nas_nn.MutableMultiheadAttention: _count_mhattn,

    nas_nn.LayerChoice: _count_layerchoice,
    nas_nn.Repeat: _count_repeat,
}
