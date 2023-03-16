# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from nni.mutable import Categorical, Mutable, Sample, SampleValidationError, MutableExpression, label_scope, auto_label
from nni.nas.nn.pytorch import Repeat, recursive_freeze

from .base import BaseSuperNetModule
from .differentiable import DifferentiableMixedLayer, DifferentiableMixedInput
from ._expression_utils import traverse_all_options

__all__ = ['ProxylessMixedLayer', 'ProxylessMixedInput', 'ProxylessMixedRepeat', 'suppress_already_mutated']


def _detach_tensor(tensor: Any, requires_grad: bool = False) -> Any:
    """Recursively detach all the tensors."""
    if isinstance(tensor, (list, tuple)):
        return tuple(_detach_tensor(t, requires_grad) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: _detach_tensor(v, requires_grad) for k, v in tensor.items()}
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if requires_grad:
            tensor.requires_grad_()
        return tensor
    else:
        return tensor


def _iter_tensors(tensor: Any) -> Any:
    """Recursively iterate over all the tensors.

    This is kept for complex outputs (like dicts / lists).
    However, complex outputs are not supported by PyTorch backward hooks yet.
    """
    if isinstance(tensor, torch.Tensor):
        yield tensor
    elif isinstance(tensor, (list, tuple)):
        for t in tensor:
            yield from _iter_tensors(t)
    elif isinstance(tensor, dict):
        for t in tensor.values():
            yield from _iter_tensors(t)


def _pack_as_tuple(tensor: Any) -> tuple:
    """Return a tuple of tensor with only one element if tensor it's not a tuple."""
    if isinstance(tensor, (tuple, list)):
        for t in tensor:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f'All elements in the tuple must be of the same type (tensor), but got {type(t)}')
        return tuple(tensor)
    elif isinstance(tensor, torch.Tensor):
        return (tensor,)
    else:
        raise TypeError(f'Unsupported type {type(tensor)}')


def _unpack_tuple(tensor: tuple) -> Any:
    """Return a single element if a single-element tuple. Otherwise a tuple."""
    if len(tensor) == 1:
        return tensor[0]
    else:
        return tensor


def element_product_sum(tensor1: tuple[torch.Tensor, ...], tensor2: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Compute the sum of all the element-wise product."""
    assert len(tensor1) == len(tensor2), 'The number of tensors must be the same.'
    # Skip zero gradients
    ret = [torch.sum(t1 * t2) for t1, t2 in zip(tensor1, tensor2) if t1 is not None and t2 is not None]
    if not ret:
        return torch.tensor(0)
    if len(ret) == 1:
        return ret[0]
    return cast(torch.Tensor, sum(ret))


class _ProxylessFunction(torch.autograd.Function):
    """
    Compute the gradient of ``arch_alpha``,
    given the sampled index, inputs and outputs of the layer, and a function to
    compute the output of the layer conditioned on any other index.

    The forward of this function is merely returning a "copy" of the given input.
    This is different from the
    `official implementation <https://github.com/mit-han-lab/proxylessnas/blob/9cdd0791/search/modules/mix_op.py>`,
    where the output is computed within this function.

    Things tried but NOT WORKING:

    1. Use ``full_backward_hook`` instead of ``autograd.Function``.
       Since ``full_backward_hook`` is not intended for computing gradients for parameters,
       The gradients get overridden when another loss is added (e.g., latency loss).
    2. Computing the output in ``autograd.Function`` like the official impl does.
       This requires calling either ``autograd.grad`` (doesn't work when nested),
       or ``autograd.backward`` in the backward of the Function class.
       The gradients within the candidates (like Linear, or Conv2d) will computed inside this function.
       This is problematic with DDP, because either (i) with ``find_unused_parameters=True``,
       the parameters within the candidates are considered unused (because they are not found on the autograd graph),
       and raises error when they receive a gradient (something like "another gradient received after ready"), or
       (ii) with ``find_unused_parameters=False``, DDP complains about some candidate paths having no gradients at all.

    Therefore, the only way to make it work is to write a function::

        func(alpha, input, output) = output

    From the outside of this box, ``func`` echoes the output. But inside, there is gradient going on.
    When back-propagation, ``alpha`` will receive a computed gradient.
    ``input`` receives none because it didn't participate in the computation
    (but only needs to be saved for computing gradients).
    ``output`` receives the gradient that ``func`` receives.
    """

    @staticmethod
    def forward(ctx, forward_path_func, sample_index, num_inputs,
                softmax, arch_alpha, *layer_input_output):
        ctx.forward_path_func = forward_path_func
        ctx.sample_index = sample_index
        # First num_inputs are inputs, the rest of outputs.
        ctx.num_inputs = num_inputs
        ctx.softmax = softmax

        ctx.save_for_backward(arch_alpha, *layer_input_output)

        layer_output = layer_input_output[num_inputs:]

        # Why requires_grad: This function can be considered as an operator of arch_alpha + some inputs.
        #                    It is a differentiable function. The results must require grad.
        # Why detach: So that requires_grad flag is not directly put onto input.
        # Why unpack_tuple: Output is packed to tuple when sending to this forward.
        #                   Need to restore the original object structure.
        # NOTE: This could be potentially troublesome if users actually returned a single-ele tuple.
        return _unpack_tuple(_detach_tensor(layer_output, requires_grad=True))

    @staticmethod
    def backward(ctx, *grad_output):
        softmax, sample_index, forward_path_func, num_inputs = \
            ctx.softmax, ctx.sample_index, ctx.forward_path_func, ctx.num_inputs

        if ctx.needs_input_grad[0]:
            # arch_alpha requires gradient. To improve efficiency.
            grads = None
        else:
            arch_alpha, *layer_input_output = ctx.saved_tensors
            layer_input = layer_input_output[:num_inputs]
            layer_output = layer_input_output[num_inputs:]

            # binary_grads is the gradient of binary gates.
            # Binary gates is a one-hot tensor where 1 is on the sampled index, and others are 0.
            # By chain rule, it's gradient is grad_output times the layer_output (of the corresponding path).

            binary_grads = torch.zeros_like(arch_alpha)

            with torch.no_grad():
                # Compute binary grads.
                for k in range(len(binary_grads)):
                    if k != sample_index:
                        out_k = forward_path_func(k, *layer_input)  # type: ignore
                    else:
                        out_k = layer_output

                    # NOTE: One limitation here is that out_k can't be complex objects like dict.
                    binary_grads[k] = element_product_sum(
                        _pack_as_tuple(out_k),  # In case out_k is a single tensor
                        grad_output             # Assuming grad_output is not None
                    )

                # Compute the gradient of the arch_alpha, based on binary_grads.
                grads = torch.zeros_like(arch_alpha)
                probs = softmax(arch_alpha)
                for i in range(len(arch_alpha)):
                    for j in range(len(arch_alpha)):
                        # Arch alpha's gradients are accumulated for all backwards through this layer.
                        grads[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])

        return (
            None, None, None, None,     # No gradients for the constants
            grads,                      # arch_alpha
            *([None] * num_inputs),     # We only use the inputs. We know nothing about the inputs.
            *grad_output                # Pass-through
        )


def suppress_already_mutated(module, name, memo, mutate_kwargs) -> bool | None:
    # ProxylessMixedRepeat will create MixedLayer on its own.
    # The created layer should NOT be mutated again.

    if isinstance(module, (ProxylessMixedLayer, ProxylessMixedInput, ProxylessMixedRepeat)):
        return True

    return None  # Skip this hook.


class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than compute the weighted sum.

    Currently the input and output of the candidate layers can only be tensors or tuple of tensors.
    They can't be dict, list or any complex types, or non-tensors (including none).
    """

    _arch_parameter_names = ['_arch_alpha']

    def __init__(self, paths: dict[str, nn.Module], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(paths, alpha, softmax, label)
        # Binary gates should be created here, but it's not because it's never used in the forward pass.
        # self._binary_gates = nn.Parameter(torch.zeros(len(paths)))

        # like sampling-based methods, it has a ``_sampled``.
        self._sampled: str | int | None = None
        self._sample_idx: int | None = None

    def forward(self, *args, **kwargs):
        """Forward pass of one single path."""
        if self._sample_idx is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        if kwargs:
            raise ValueError(f'kwargs is not supported yet in {self.__class__.__name__}.')
        result = self.forward_path(self._sample_idx, *args, **kwargs)
        return _ProxylessFunction.apply(
            self.forward_path, self._sample_idx, len(args), self._softmax,
            self._arch_alpha, *args, *_pack_as_tuple(result)
        )

    def forward_path(self, index, *args, **kwargs):
        return self[self.names[index]](*args, **kwargs)

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
            self._sample_idx = self.names.index(self._sampled)
        else:
            probs = self._softmax(self._arch_alpha)
            self._sample_idx = int(torch.multinomial(probs, 1)[0].item())
            self._sampled = self.names[self._sample_idx]

        return {self.label: self._sampled}

    def export(self, memo):
        """Same as :meth:`resample`."""
        return self.resample(memo)


class ProxylessMixedInput(DifferentiableMixedInput):
    """Proxyless version of differentiable input choice.
    See :class:`ProxylessMixedLayer` for implementation details.
    """

    _arch_parameter_names = ['_arch_alpha', '_binary_gates']

    def __init__(self, n_candidates: int, n_chosen: int | None, alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(n_candidates, n_chosen, alpha, softmax, label)

        # We only support choosing a particular one in forward.
        # Nevertheless, we rank the score and export the tops in export.
        self._sampled: list[int] | None = None

    def forward(self, inputs):
        """Choose one single input."""
        if self._sampled is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        result = self.forward_path(self._sampled[0], *inputs)
        return _ProxylessFunction.apply(
            self.forward_path, self._sampled[0], len(inputs), self._softmax,
            self._arch_alpha, *inputs, result
        )

    def forward_path(self, index, *inputs):
        return inputs[index]

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self._softmax(self._arch_alpha)
            # TODO: support real n_chosen is None
            n_chosen = self.n_chosen or 1
            sample = torch.multinomial(probs, n_chosen).cpu().numpy().tolist()
            self._sampled = sample

        return {self.label: self._sampled}

    def export(self, memo):
        """Same as :meth:`resample`."""
        return self.resample(memo)


class ProxylessMixedRepeat(Repeat, BaseSuperNetModule):
    """ProxylessNAS converts repeat to a sequential blocks of layer choices between
    the original block and an identity layer.

    Only pure categorical depth choice is supported.
    If the categorical choices are not consecutive integers, the constraint will only be considered at export.
    """

    depth_choice: Categorical[int]

    def __init__(self, blocks: list[nn.Module], depth: Categorical[int]):
        super().__init__(blocks, depth)
        assert isinstance(depth, Categorical)
        assert len(blocks) == self.max_depth
        for d in range(self.min_depth, self.max_depth):
            block = blocks[d]
            assert isinstance(block, ProxylessMixedLayer)
            assert len(block._arch_alpha) == 2

    def resample(self, memo):
        """Resample each individual depths."""
        if self.depth_choice.label in memo:
            return {}
        depth = self.min_depth
        for d in range(self.min_depth, self.max_depth):
            layer = self.blocks[d]
            assert isinstance(layer, ProxylessMixedLayer)
            # The depth-related choices must be sampled here.
            memo.pop(layer.label, None)
            sample = layer.resample(memo)
            memo.update(sample)
            depth += int(memo[layer.label])
        return {self.depth_choice.label: depth}

    def export(self, memo):
        """Return the most likely to be chosen depth choice."""
        sample = {}
        for _ in range(1000):
            sample = self.resample(memo)
            if sample[self.depth_choice.label] in self.depth_choice.values:
                return sample
        # Sampling failed after 1000 retries. Return an arbitrary one.
        return sample

    def export_probs(self, memo):
        """Compute the probability of each depth choice gets chosen."""
        if self.depth_choice.label in memo:
            return {}
        categoricals: list[Categorical] = []
        weights: dict[str, torch.Tensor] = {}
        for d in range(self.min_depth, self.max_depth):
            layer = cast(ProxylessMixedLayer, self.blocks[d])
            categoricals.append(MutableExpression.to_int(layer.choice))
            weights[layer.label] = layer._softmax(layer._arch_alpha)
        return {self.depth_choice.label: dict(
            traverse_all_options(cast(MutableExpression[int], sum(categoricals) + self.min_depth), weights)
        )}

    def check_contains(self, sample: Sample) -> SampleValidationError | None:
        # Check depth choice
        exception = self.depth_choice.check_contains(sample)
        if exception is not None:
            return exception
        depth = self.depth_choice.freeze(sample)

        # Check blocks
        for i, block in enumerate(self.blocks):
            if i < self.min_depth:
                exception = self._check_any_module_contains(block, sample, str(i))
            elif i < depth:
                assert isinstance(block, ProxylessMixedLayer)
                exception = self._check_any_module_contains(block['1'], sample, str(i))
            else:
                break
        return None

    def freeze(self, sample: Sample) -> nn.Sequential:
        self.validate(sample)
        depth = self.depth_choice.freeze(sample)
        blocks = []
        for i, block in enumerate(self.blocks):
            if i < self.min_depth:
                blocks.append(recursive_freeze(block, sample)[0])
            elif i < depth:
                assert isinstance(block, ProxylessMixedLayer)
                blocks.append(recursive_freeze(block['1'], sample)[0])
            else:
                break
        return nn.Sequential(*blocks)

    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if type(module) == Repeat and isinstance(module.depth_choice, Mutable):  # Repeat and depth is mutable
            module = cast(Repeat, module)
            if not isinstance(module.depth_choice, Categorical):
                raise ValueError(f'The depth choice must be a straightforward categorical, but got {module.depth_choice}')
            blocks: list[nn.Module] = []
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))

            with label_scope(module.depth_choice.label):
                for i, block in enumerate(module.blocks):
                    if i < module.min_depth:
                        blocks.append(block)
                    else:
                        # Creating local labels here.
                        label = auto_label(f'in_repeat_{i}')
                        if label in memo:
                            alpha = memo[label]
                        else:
                            alpha = nn.Parameter(torch.randn(2) * 1E-3)
                            memo[label] = alpha
                        candidates = {
                            # Has to be strings here because when using dict as layer choice parameter,
                            # the keys of the dict must be string.
                            '0': nn.Identity(),   # skip
                            '1': block,           # choose
                        }
                        blocks.append(ProxylessMixedLayer(candidates, alpha, softmax, label))

            return cls(blocks, module.depth_choice)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
