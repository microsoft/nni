import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDifferentiableOpChoice(nn.Module):
    def __init__(self, ops):
        super(SimpleDifferentiableOpChoice, self).__init__()
        self.ops = nn.ModuleList(ops)
        self._arch_weight = nn.Parameter(torch.randn(len(ops)) * 1E-3)

    def forward(self, *args):
        weights = F.softmax(self._arch_weight, dim=-1)
        results = torch.stack([op(*args) for op in self.ops], 0)
        weights_shape = [-1] + [1] * (len(results.size()) - 1)
        return torch.sum(weights.view(*weights_shape) * results, 0)


class SimpleDifferentiableTensorChoice(nn.Module):
    def __init__(self, num_candidates):
        super(SimpleDifferentiableTensorChoice, self).__init__()
        self.num_candidates = num_candidates
        self._arch_weight = nn.Parameter(torch.randn(num_candidates) * 1E-3)

    def forward(self, tensors):
        weights = F.softmax(self._arch_weight, dim=-1)
        results = torch.stack(tensors, 0)
        weights_shape = [-1] + [1] * (len(results.size()) - 1)
        return torch.sum(weights.view(*weights_shape) * results, 0)


class ArchGradientFunction(torch.autograd.Function):
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


class ProxylessNASMixedOp(nn.Module):
    def __init__(self, ops):
        super(ProxylessNASMixedOp, self).__init__()
        self.ops = nn.ModuleList(ops)
        self._arch_parameters = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self.sampled = None

    def forward(self, *args):
        def run_function(ops, active_id):
            def forward(_x):
                return ops[active_id](_x)
            return forward

        def backward_function(ops, active_id, binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(len(ops)):
                        if k != active_id:
                            out_k = ops[k](_x.data)
                        else:
                            out_k = _output.data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads
            return backward

        assert len(args) == 1
        x = args[0]
        return ArchGradientFunction.apply(
            x, self._binary_gates, run_function(self.ops, self.sampled),
            backward_function(self.ops, self.sampled, self._binary_gates)
        )

    def resample(self):
        probs = F.softmax(self._arch_parameters, dim=-1)
        sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.data[sample] = 1.0

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self._arch_parameters.grad is None:
                self._arch_parameters.grad = torch.zeros_like(self._arch_parameters.data)
            probs = F.softmax(self._arch_parameters, dim=-1)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self._arch_parameters.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])
