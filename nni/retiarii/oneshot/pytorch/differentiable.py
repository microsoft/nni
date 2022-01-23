# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from .base_lightning import BaseOneShotLightningModule
from .darts import DartsInputChoice, DartsLayerChoice


class DartsModule(BaseOneShotLightningModule):
    """
    The DARTS model. In each iteration, there are 2 training phases. The phase 1 is architecture
    step, in which the model parameters are frozen and the alphas are optimized. The phase 2 is model
    step, in wchich the alphas are frozen and the model parameters are optimized. In this step, the
    out put of each choice are sumed up with respect to their alpha value. The result of DARTS
    is argmax in alpha.
    The DARTS Model should be trained with ParallelTraiValDataloader in nn.retiarii.oneshot.pytorch.utils.
    See base class for more attributes.

    Parameters
    ----------
    base_model : pl.LightningModule
        The module in evaluators in nni.retiarii.evaluator.lightning. User defined model
        is wrapped by base_model, and base_model will be wrapped by this model.
    custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
        The custom xxxChoice replace method. Keys should be xxxChoice type and values should
        return an nn.module. This custom replace dict will override the default replace
        dict of each NAS method.

    Reference
    ----------
    .. [darts] H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable Architecture Search,” presented at the
        International Conference on Learning Representations, Sep. 2018. Available: https://openreview.net/forum?id=S1eYHoC5FX
    """
    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # grad manually, only 1 architecture optimizer for darts
        opts = self.optimizers()
        arc_optim = opts[0]
        w_optim = opts[1:]

        # ParallelTrainValDataLoader will yield both train and val data in a batch
        trn_batch, val_batch = batch

        # phase 1: architecture step
        # The resample_architecture hook is kept for some following darts-based NAS
        # methods such as proxyless. See code of those methods for details.
        self.resample_architecture()
        arc_optim.zero_grad()
        arc_step_loss = self._extract_user_loss(val_batch, 2 * batch_idx)
        self.manual_backward(arc_step_loss)
        self.finalize_grad()
        arc_optim.step()

        # phase 2: model step
        self.resample_architecture()
        for opt in w_optim:
            opt.zero_grad()
        w_step_loss = self._extract_user_loss(trn_batch, 2 * batch_idx + 1)
        self.manual_backward(w_step_loss)
        for opt in w_optim:
            opt.step()
        return w_step_loss

    def resample_architecture(self):
        """
        Hook kept for darts-based methods. Some following works resample the architecture to
        reduce the memory consumption during training to fit the method to bigger models. Details
        are provided in code of those algorithms.
        """
        pass

    def finalize_grad(self):
        # Note: This hook is currently kept for Proxyless NAS.
        pass

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : DartsLayerChoice,
            InputChoice : DartsInputChoice
        }

    def configure_architecture_optimizers(self):
        # The alpha in DartsXXXChoices are the architecture parameters of DARTS. They share one optimizer.
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), 3.e-4, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        return ctrl_optim


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


class ProxylessLayerChoice(nn.Module):
    def __init__(self, ops):
        super(ProxylessLayerChoice, self).__init__()
        self.ops = nn.ModuleList(ops)
        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self.sampled = None

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

    def resample(self):
        probs = F.softmax(self.alpha, dim=-1)
        sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            probs = F.softmax(self.alpha, dim=-1)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self.alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])

    def export(self):
        return torch.argmax(self.alpha).item()

    def export_prob(self):
        return F.softmax(self.alpha, dim=-1)


class ProxylessInputChoice(nn.Module):
    def __init__(self, input_choice):
        super().__init__()
        self.ops = nn.ModuleList(input_choice)
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(input_choice.n_candidates) * 1E-3)
        self.sampled = None

    def forward(self, inputs):
        if self.training:
            def run_function(active_sample):
                return lambda x: x[active_sample]

            def backward_function(binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(self.num_input_candidates):
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

    def resample(self, sample=None):
        if sample is None:
            probs = F.softmax(self.alpha, dim=-1)
            sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0
        return self.sampled

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            probs = F.softmax(self.alpha, dim=-1)
            for i in range(self.num_input_candidates):
                for j in range(self.num_input_candidates):
                    self.alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])


class ProxylessModule(DartsModule):
    """
    The Proxyless Module. This is a darts-based method. What it differs from darts is that it resample
    the architecture according to alphas to select only one path a time to reduce memory consumption.
    The Proxyless Model should be trained with ParallelTraiValDataloader in nn.retiarii.oneshot.pytorch.utils.
    See base class for more attributes.

    Reference
    ----------
    [proxyless] H. Cai, L. Zhu, and S. Han, “ProxylessNAS: Direct Neural Architecture Search on Target
        Task and Hardware,” presented at the International Conference on Learning Representations,
        Sep. 2018. Available: https://openreview.net/forum?id=HylVB3AqYm

    """
    def configure_architecture_optimizers(self):
        ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], lr=3.e-4,
                                           weight_decay=0, betas=(0, 0.999), eps=1e-8)
        return ctrl_optim

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : ProxylessLayerChoice,
            InputChoice : ProxylessInputChoice
        }

    def resample_architecture(self):
        for _, m in self.nas_modules:
            m.resample()

    def finalize_grad(self):
        for _, m in self.nas_modules:
            m.finalize_grad()


class SNASLayerChoice(nn.Module):
    def __init__(self, layer_choice, temp = 1):
        super().__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
        self._temp = temp

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.one_hot, -1).view(*alpha_shape), 0)

    def resample(self):
        # gumble soft-max
        log_alpha = torch.log(self.alpha)
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        self.one_hot = softmax((log_alpha -(-u.log()).log()) / self._temp)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]

    def r_forward(self, x):
        return self.op_choices.values()[0](x)


class SNASInputChoice(nn.Module):
    def __init__(self, input_choice, temp = 1):
        super().__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1
        self._temp = temp

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.one_hot, -1).view(*alpha_shape), 0)

    def resample(self):
        # gumble soft-max
        log_alpha = torch.log(self.alpha)
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        self.one_hot = softmax((log_alpha -(-u.log()).log()) / self._temp)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]

    def r_forward(self, x):
        return super().r_forward(x)


class SNASModule(DartsModule):
    """
    The SNAS Module. This is a darts-based method. It uses gumble-softmax to simulate a one-hot distribution to
    select only one path a time. The SNAS Module should be trained with ParallelTrainValDataLoader in
    nn.retiarii.oneshot.utils.
    See base class for more attributes.

    Parameters
    ----------
    base_model : pl.LightningModule
        The module in evaluators in nni.retiarii.evaluator.lightning. User defined model
        is wrapped by base_model, and base_model will be wrapped by this model.
    gumble_temperature : float
        The temperature used in gumble-softmax. See [snas] for details.
    custom_replace_dict : Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]], default = None
        The custom xxxChoice replace method. Keys should be xxxChoice type and values should
        return an nn.module. This custom replace dict will override the default replace
        dict of each NAS method.

    Reference
    ----------
    ..[snas] S. Xie, H. Zheng, C. Liu, and L. Lin, “SNAS: stochastic neural architecture search,” presented at the
        International Conference on Learning Representations, Sep. 2018. Available: https://openreview.net/forum?id=rylqooRqK7
    """
    def __init__(self, base_model, gumble_temperature = 1., custom_replace_dict=None):
        self.temp = gumble_temperature
        super().__init__(base_model, custom_replace_dict)

    def configure_architecture_optimizers(self):
        ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], lr=3.e-4,
                                           weight_decay=0, betas=(0, 0.999), eps=1e-8)
        return ctrl_optim

    @property
    def default_replace_dict(self):
        return {
            LayerChoice : lambda m : SNASLayerChoice(m, self.temp),
            InputChoice : lambda m : SNASInputChoice(m, self.temp)
        }

    def resample_architecture(self):
        for _, m in self.nas_modules:
            m.resample()
