"""Implementation of ProxylessNAS: a hyrbid approach between differentiable and sampling.
The support remains limited. Known limitations include:

- No support for multiple arguments in forward.
- No support for mixed-operation (value choice).
- The code contains duplicates. Needs refactor.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .differentiable import DifferentiableMixedLayer, DifferentiableMixedInput


class _ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = x.detach()
        detached_x.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None


class ProxylessMixedLayer(DifferentiableMixedLayer):
    """Proxyless version of differentiable mixed layer.
    It resamples a single-path every time, rather than go through the softmax.
    """

    def __init__(self, paths: List[Tuple[str, nn.Module]], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(paths, alpha, softmax, label)
        self._binary_gates = nn.Parameter(torch.randn(len(paths)) * 1E-3)

        # like sampling-based methods, it has a ``_sampled``.
        self._sampled: Optional[str] = None

    def forward(self, *args, **kwargs):
        if self.training:
            def run_function(ops, active_id, **kwargs):
                def forward(_x):
                    return ops[active_id](_x, **kwargs)
                return forward

            def backward_function(ops, active_id, binary_gates, **kwargs):
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

            assert len(args) == 1
            x = args[0]
            return _ArchGradientFunction.apply(
                x, self._binary_gates, run_function(self.ops, self.sampled, **kwargs),
                backward_function(self.ops, self.sampled, self._binary_gates, **kwargs)
            )

        return super().forward(*args, **kwargs)

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self.softmax(self._arch_alpha)
            sample = torch.multinomial(probs, 1)[0].item()
            self._sampled = sample

        # set binary gates
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

        return {self.label: self._sampled}

    def export(self, memo):
        """Chose the argmax if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: torch.argmax(self._arch_alpha).item()}

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self._arch_alpha.grad is None:
                self._arch_alpha.grad = torch.zeros_like(self._arch_alpha.data)
            probs = self.softmax(self._arch_alpha)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self._arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])


class ProxylessInputChoice(DifferentiableMixedInput):
    """Proxyless version of differentiable input choice.
    See :class:`ProxylessLayerChoice` for implementation details.
    """

    def __init__(self, n_candidates: int, n_chosen: Optional[int], alpha: torch.Tensor, softmax: nn.Module, label: str):
        super().__init__(n_candidates, n_chosen, alpha, softmax, label)
        self._binary_gates = nn.Parameter(torch.randn(n_candidates) * 1E-3)
        self._sampled = None

    def forward(self, inputs):
        if self.training:
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
            return _ArchGradientFunction.apply(
                inputs, self._binary_gates, run_function(self.sampled),
                backward_function(self._binary_gates)
            )

        return super().forward(inputs)

    def resample(self, memo):
        """Sample one path based on alpha if label is not found in memo."""
        if self.label in memo:
            self._sampled = memo[self.label]
        else:
            probs = self.softmax(self._arch_alpha)
            sample = torch.multinomial(probs, 1)[0].item()
            self._sampled = sample

        # set binary gates
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

        return {self.label: self._sampled}

    def export(self, memo):
        """Chose the argmax if label isn't found in memo."""
        if self.label in memo:
            return {}  # nothing new to export
        return {self.label: torch.argmax(self._arch_alpha).item()}

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self._arch_alpha.grad is None:
                self._arch_alpha.grad = torch.zeros_like(self._arch_alpha.data)
            probs = self.softmax(self._arch_alpha)
            for i in range(self.n_candidates):
                for j in range(self.n_candidates):
                    self._arch_alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])
