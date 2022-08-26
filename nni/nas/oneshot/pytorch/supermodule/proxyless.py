# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""

from __future__ import annotations

from typing import Any, Tuple, Union, cast

import torch
import torch.nn as nn

from .differentiable import DifferentiableMixedLayer, DifferentiableMixedInput

__all__ = ['ProxylessMixedLayer', 'ProxylessMixedInput']


def _detach_tensor(tensor: Any) -> Any:
    """Recursively detach all the tensors."""
    if isinstance(tensor, (list, tuple)):
        return tuple(_detach_tensor(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: _detach_tensor(v) for k, v in tensor.items()}
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach()
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
        return tuple(tensor)
    return (tensor,)


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


class ProxylessContext:

    def __init__(self, arch_alpha: torch.Tensor, softmax: nn.Module) -> None:
        self.arch_alpha = arch_alpha
        self.softmax = softmax

        # When a layer is called multiple times, the inputs and outputs are saved in order.
        # In backward propagation, we assume that they are used in the reversed order.
        self.layer_input: list[Any] = []
        self.layer_output: list[Any] = []
        self.layer_sample_idx: list[int] = []

    def clear_context(self) -> None:
        self.layer_input = []
        self.layer_output = []
        self.layer_sample_idx = []

    def save_forward_context(self, layer_input: Any, layer_output: Any, layer_sample_idx: int):
        self.layer_input.append(_detach_tensor(layer_input))
        self.layer_output.append(_detach_tensor(layer_output))
        self.layer_sample_idx.append(layer_sample_idx)

    def backward_hook(self, module: nn.Module,
                      grad_input: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                      grad_output: Union[Tuple[torch.Tensor, ...], torch.Tensor]) -> None:
        # binary_grads is the gradient of binary gates.
        # Binary gates is a one-hot tensor where 1 is on the sampled index, and others are 0.
        # By chain rule, it's gradient is grad_output times the layer_output (of the corresponding path).

        binary_grads = torch.zeros_like(self.arch_alpha)

        # Retrieve the layer input/output in reverse order.
        if not self.layer_input:
            raise ValueError('Unexpected backward call. The saved context is empty.')
        layer_input = self.layer_input.pop()
        layer_output = self.layer_output.pop()
        layer_sample_idx = self.layer_sample_idx.pop()

        with torch.no_grad():
            # Compute binary grads.
            for k in range(len(binary_grads)):
                if k != layer_sample_idx:
                    args, kwargs = layer_input
                    out_k = module.forward_path(k, *args, **kwargs)  # type: ignore
                else:
                    out_k = layer_output

                # FIXME: One limitation here is that out_k can't be complex objects like dict.
                # I think it's also a limitation of backward hook.
                binary_grads[k] = element_product_sum(
                    _pack_as_tuple(out_k),  # In case out_k is a single tensor
                    _pack_as_tuple(grad_output)
                )

            # Compute the gradient of the arch_alpha, based on binary_grads.
            if self.arch_alpha.grad is None:
                self.arch_alpha.grad = torch.zeros_like(self.arch_alpha)
            probs = self.softmax(self.arch_alpha)
            for i in range(len(self.arch_alpha)):
                for j in range(len(self.arch_alpha)):
                    # Arch alpha's gradients are accumulated for all backwards through this layer.
                    self.arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])


class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than go through the softmax.
    """

    _arch_parameter_names = ['_arch_alpha']

    def __init__(self, paths: list[tuple[str, nn.Module]], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(paths, alpha, softmax, label)
        # Binary gates should be created here, but it's not because it's never used in the forward pass.
        # self._binary_gates = nn.Parameter(torch.zeros(len(paths)))

        # like sampling-based methods, it has a ``_sampled``.
        self._sampled: str | None = None
        self._sample_idx: int | None = None

        # arch_alpha could be shared by multiple layers,
        # but binary_gates is owned by the current layer.
        self.ctx = ProxylessContext(alpha, softmax)
        self.register_full_backward_hook(self.ctx.backward_hook)

    def forward(self, *args, **kwargs):
        """Forward pass of one single path."""
        if self._sample_idx is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        output = self.forward_path(self._sample_idx, *args, **kwargs)
        self.ctx.save_forward_context((args, kwargs), output, self._sample_idx)
        return output

    def forward_path(self, index, *args, **kwargs):
        return getattr(self, self.op_names[index])(*args, **kwargs)

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
            self._sample_idx = self.op_names.index(self._sampled)
        else:
            probs = self._softmax(self._arch_alpha)
            self._sample_idx = int(torch.multinomial(probs, 1)[0].item())
            self._sampled = self.op_names[self._sample_idx]

        self.ctx.clear_context()

        return {self.label: self._sampled}


class ProxylessMixedInput(DifferentiableMixedInput):
    """Proxyless version of differentiable input choice.
    See :class:`ProxylessMixedLayer` for implementation details.
    """

    _arch_parameter_names = ['_arch_alpha', '_binary_gates']

    def __init__(self, n_candidates: int, n_chosen: int | None, alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(n_candidates, n_chosen, alpha, softmax, label)

        # We only support choosing a particular one here.
        # Nevertheless, we rank the score and export the tops in export.
        self._sampled: int | None = None
        self.ctx = ProxylessContext(alpha, softmax)
        self.register_full_backward_hook(self.ctx.backward_hook)

    def forward(self, inputs):
        """Choose one single input."""
        if self._sampled is None:
            raise RuntimeError('resample() needs to be called before fprop.')
        output = self.forward_path(self._sampled, inputs)
        self.ctx.save_forward_context(((inputs,), {}), output, self._sampled)
        return output

    def forward_path(self, index, inputs):
        return inputs[index]

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self._softmax(self._arch_alpha)
            sample = torch.multinomial(probs, 1)[0].item()
            self._sampled = int(sample)

        self.ctx.clear_context()

        return {self.label: self._sampled}
