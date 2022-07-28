# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .differentiable import DifferentiableMixedLayer, DifferentiableMixedInput

__all__ = ['ProxylessMixedLayer', 'ProxylessMixedInput']


class ProxylessContext:

    def __init__(self, arch_alpha: torch.Tensor, binary_gates: torch.Tensor) -> None:
        self.arch_alpha = arch_alpha
        self.binary_gates = binary_gates

        self.output = None

    def forward_hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        # Save the output to be used in gradients.
        self.output = output.data

    def backward_hook(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
        def backward(_x, _output, grad_output):
            binary_grads = torch.zeros_like(binary_gates.data)
            with torch.no_grad():
                for k in range(len(ops)):
                    if k != active_id:
                        out_k = ops[k](_x.data, **kwargs)
                    else:
                        out_k = _output.data
                    grad_k = torch.sum(out_k * grad_output)
                    binary_grads[k] = grad_k
            return binary_grads
        return backward



def _proxyless_backward_hook(self, grad_input, grad_output):
    """
    A backward hook to handle the computation of binary gates and alpha in Proxyless layers.
    """



class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than go through the softmax.
    """

    _arch_parameter_names = ['_arch_alpha', '_binary_gates']

    def __init__(self, paths: list[tuple[str, nn.Module]], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(paths, alpha, softmax, label)
        self._binary_gates = nn.Parameter(torch.zeros(len(paths)))

        # like sampling-based methods, it has a ``_sampled``.
        self._sampled: str | None = None
        self._sample_idx: int | None = None

    def forward(self, *args, **kwargs):
        """Forward pass of one single path."""
        return getattr(self, self.op_names[self._sample_idx])(*args, **kwargs)

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
            self._sample_idx = self.op_names.index(self._sampled)
        else:
            probs = self._softmax(self._arch_alpha)
            self._sample_idx = int(torch.multinomial(probs, 1)[0].item())
            self._sampled = self.op_names[self._sample_idx]
        print(self._sampled, self._sample_idx)

        # set binary gates
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[self._sample_idx] = 1.0

        return {self.label: self._sampled}

    def export(self, memo):
        """Chose the argmax if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: self.op_names[int(torch.argmax(self._arch_alpha).item())]}

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        assert binary_grads is not None
        with torch.no_grad():
            if self._arch_alpha.grad is None:
                self._arch_alpha.grad = torch.zeros_like(self._arch_alpha.data)
            probs = self._softmax(self._arch_alpha)
            for i in range(len(self._arch_alpha)):
                for j in range(len(self._arch_alpha)):
                    self._arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])


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
