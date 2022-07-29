# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""

from __future__ import annotations

from typing import cast, Any

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
    """Recursively iterate over all the tensors."""
    if isinstance(tensor, (list, tuple)):
        yield from (_iter_tensors(t) for t in tensor)
    elif isinstance(tensor, dict):
        yield from (_iter_tensors(t) for t in tensor.values())
    elif isinstance(tensor, torch.Tensor):
        yield tensor


def element_product_sum(tensor1: tuple[torch.Tensor, ...], tensor2: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Compute the sum of all the element-wise product."""
    assert len(tensor1) == len(tensor2), 'The number of tensors must be the same.'
    ret = [torch.sum(t1 * t2) for t1, t2 in zip(tensor1, tensor2)]
    if len(ret) == 1:
        return ret[0]
    return sum(ret)


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

    def backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
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
                    out_k = module.forward_path(*args, **kwargs)
                else:
                    out_k = layer_output
                binary_grads[k] = element_product_sum(tuple(_iter_tensors(out_k)), grad_output)

            # Compute the gradient of the arch_alpha, based on binary_grads.
            if self.arch_alpha.grad is None:
                self.arch_alpha.grad = torch.zeros_like(self.arch_alpha)
            probs = self.softmax(self.arch_alpha)
            for i in range(len(self.arch_alpha)):
                for j in range(len(self.arch_alpha)):
                    self.arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])


class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than go through the softmax.
    """

    _arch_parameter_names = ['_arch_alpha']

    def __init__(self, paths: list[tuple[str, nn.Module]], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(paths, alpha, softmax, label)
        self._binary_gates = nn.Parameter(torch.zeros(len(paths)))

        # like sampling-based methods, it has a ``_sampled``.
        self._sampled: str | None = None
        self._sample_idx: int | None = None

        # arch_alpha could be shared by multiple layers,
        # but binary_gates is owned by the current layer.
        self.ctx = ProxylessContext(alpha, softmax)
        self.register_full_backward_hook(self.ctx.backward_hook)

    def forward(self, *args, **kwargs):
        """Forward pass of one single path."""
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
    See :class:`ProxylessLayerChoice` for implementation details.
    """

    _arch_parameter_names = ['_arch_alpha', '_binary_gates']

    def __init__(self, n_candidates: int, n_chosen: int | None, alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(n_candidates, n_chosen, alpha, softmax, label)
        self._binary_gates = nn.Parameter(torch.zeros(n_candidates))
        self._sampled: int | None = None

    def forward(self, inputs):
        def run_function(active_sample):
            return lambda x: x[active_sample]

        def backward_function(binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(self.n_candidates):
                        out_k = _x[k].data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads
            return backward

        inputs = torch.stack(inputs, 0)
        assert self._sampled is not None, 'Need to call resample() before running fprop.'

        return _ArchGradientFunction.apply(
            inputs, self._binary_gates, run_function(self._sampled),
            backward_function(self._binary_gates)
        )

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self._softmax(self._arch_alpha)
            sample = torch.multinomial(probs, 1)[0].item()
            self._sampled = int(sample)

        # set binary gates
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[cast(int, self._sampled)] = 1.0

        return {self.label: self._sampled}

    def export(self, memo):
        """Chose the argmax if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: torch.argmax(self._arch_alpha).item()}

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        assert binary_grads is not None
        with torch.no_grad():
            if self._arch_alpha.grad is None:
                self._arch_alpha.grad = torch.zeros_like(self._arch_alpha.data)
            probs = self._softmax(self._arch_alpha)
            for i in range(self.n_candidates):
                for j in range(self.n_candidates):
                    self._arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])
