# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.retiarii.nn.pytorch import LayerChoice, InputChoice, Conv2d, BatchNorm2d

from .superlayer.differentiable import DifferentiableBatchNorm2d, DifferentiableSuperConv2d
from .utils import get_differentiable_valuechoice_match_and_replace, get_naive_match_and_replace
from .base_lightning import BaseOneShotLightningModule, ReplaceDictType


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse=False):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]


class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, recurse=False):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]


class DartsModule(BaseOneShotLightningModule):
    _darts_note = """
    DARTS :cite:p:`liu2018darts` algorithm is one of the most fundamental one-shot algorithm.

    DARTS repeats iterations, where each iteration consists of 2 training phases.
    The phase 1 is architecture step, in which model parameters are frozen and the architecture parameters are trained.
    The phase 2 is model step, in which architecture parameters are frozen and model parameters are trained.

    The current implementation is for DARTS in first order. Second order (unrolled) is not supported yet.

    {{module_notes}}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(base_params=BaseOneShotLightningModule._custom_replace_dict_note)

    __doc__ = _darts_note.format(
        module_notes='The DARTS Module should be trained with :class:`nni.retiarii.oneshot.utils.InterleavedTrainValDataLoader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    def __init__(self, inner_module: pl.LightningModule,
                 custom_replace_dict: Optional[ReplaceDictType] = None,
                 arc_learning_rate: float = 3.0E-4):
        super().__init__(inner_module, custom_replace_dict=custom_replace_dict)
        self.arc_learning_rate = arc_learning_rate

    def training_step(self, batch, batch_idx):
        # grad manually
        arc_optim = self.architecture_optimizers

        # The InterleavedTrainValDataLoader yields both train and val data in a batch
        trn_batch, val_batch = batch

        # phase 1: architecture step
        # The _resample hook is kept for some darts-based NAS methods like proxyless.
        # See code of those methods for details.
        self._resample()
        arc_optim.zero_grad()
        arc_step_loss = self.model.training_step(val_batch, 2 * batch_idx)
        if isinstance(arc_step_loss, dict):
            arc_step_loss = arc_step_loss['loss']
        self.manual_backward(arc_step_loss)
        self.finalize_grad()
        arc_optim.step()

        # phase 2: model step
        self._resample()
        self.call_user_optimizers('zero_grad')
        loss_and_metrics = self.model.training_step(trn_batch, 2 * batch_idx + 1)
        w_step_loss = loss_and_metrics['loss'] \
            if isinstance(loss_and_metrics, dict) else loss_and_metrics
        self.manual_backward(w_step_loss)
        self.call_user_optimizers('step')

        self.call_lr_schedulers(batch_idx)

        return loss_and_metrics

    def _resample(self):
        """
        Hook kept for darts-based methods. Some following works resample the architecture to
        reduce the memory consumption during training to fit the method to bigger models. Details
        are provided in code of those algorithms.
        """
        pass

    def finalize_grad(self):
        # Note: This hook is currently kept for Proxyless NAS.
        pass

    @staticmethod
    def match_and_replace():
        inputchoice_replace = get_naive_match_and_replace(InputChoice, DartsInputChoice)
        layerchoice_replace = get_naive_match_and_replace(LayerChoice, DartsLayerChoice)

        conv2d_valuechoice_replace = get_differentiable_valuechoice_match_and_replace(Conv2d, DifferentiableSuperConv2d)
        batch_norm2d_valuechoice_replace = get_differentiable_valuechoice_match_and_replace(BatchNorm2d, DifferentiableBatchNorm2d)

        return [inputchoice_replace, layerchoice_replace, conv2d_valuechoice_replace, batch_norm2d_valuechoice_replace]

    def configure_architecture_optimizers(self):
        # The alpha in DartsXXXChoices are the architecture parameters of DARTS. They share one optimizer.
        ctrl_params = {}
        for _, m in self.nas_modules:
            # TODO: unify layerchoice/inputchoice and valuechoice alpha
            if m.label in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.label].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.label]
            else:
                if isinstance(m.alpha, dict):
                    for k, v in m.alpha.items():
                        ctrl_params[f'{m.label}_{k}'] = v
                else:
                    ctrl_params[m.label] = m.alpha
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
        self.num_input_candidates = input_choice.n_candidates
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
    _proxyless_note = """
    Implementation of ProxylessNAS :cite:p:`cai2018proxylessnas`.
    It's a DARTS-based method that resamples the architecture to reduce memory consumption.
    Essentially, it samples one path on forward,
    and implements its own backward to update the architecture parameters based on only one path.

    {{module_notes}}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(base_params=BaseOneShotLightningModule._custom_replace_dict_note)

    __doc__ = _proxyless_note.format(
        module_notes='This module should be trained with :class:`nni.retiarii.oneshot.pytorch.utils.InterleavedTrainValDataLoader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    @staticmethod
    def match_and_replace():
        inputchoice_replace = get_naive_match_and_replace(InputChoice, ProxylessInputChoice)
        layerchoice_replace = get_naive_match_and_replace(LayerChoice, ProxylessLayerChoice)

        return [inputchoice_replace, layerchoice_replace]


    def configure_architecture_optimizers(self):
        ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], 3.e-4,
                                           weight_decay=0, betas=(0, 0.999), eps=1e-8)
        return ctrl_optim

    def _resample(self):
        for _, m in self.nas_modules:
            m.resample()

    def finalize_grad(self):
        for _, m in self.nas_modules:
            m.finalize_grad()


class SNASLayerChoice(DartsLayerChoice):
    def forward(self, *args, **kwargs):
        one_hot = F.gumbel_softmax(self.alpha, self.temp)
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        yhat = torch.sum(op_results * one_hot.view(*alpha_shape), 0)
        return yhat


class SNASInputChoice(DartsInputChoice):
    def forward(self, inputs):
        one_hot = F.gumbel_softmax(self.alpha, self.temp)
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        yhat = torch.sum(inputs * one_hot.view(*alpha_shape), 0)
        return yhat


class SnasModule(DartsModule):
    _snas_note = """
    Implementation of SNAS :cite:p:`xie2018snas`.
    It's a DARTS-based method that uses gumbel-softmax to simulate one-hot distribution.
    Essentially, it samples one path on forward,
    and implements its own backward to update the architecture parameters based on only one path.

    {{module_notes}}

    Parameters
    ----------
    {{module_params}}
    {base_params}
    gumbel_temperature : float
        The initial temperature used in gumbel-softmax.
    use_temp_anneal : bool
        If true, a linear annealing will be applied to ``gumbel_temperature``.
        Otherwise, run at a fixed temperature. See :cite:t:`xie2018snas` for details.
    min_temp : float
        The minimal temperature for annealing. No need to set this if you set ``use_temp_anneal`` False.
    arc_learning_rate : float
        Learning rate for architecture optimizer. Default: 3.0e-4
    """.format(base_params=BaseOneShotLightningModule._custom_replace_dict_note)

    __doc__ = _snas_note.format(
        module_notes='This module should be trained with :class:`nni.retiarii.oneshot.pytorch.utils.InterleavedTrainValDataLoader`.',
        module_params=BaseOneShotLightningModule._inner_module_note,
    )

    def __init__(self, inner_module,
                 custom_replace_dict: Optional[ReplaceDictType] = None,
                 arc_learning_rate: float = 3.0e-4,
                 gumbel_temperature: float = 1.,
                 use_temp_anneal: bool = False,
                 min_temp: float = .33):
        super().__init__(inner_module, custom_replace_dict, arc_learning_rate=arc_learning_rate)
        self.temp = gumbel_temperature
        self.init_temp = gumbel_temperature
        self.use_temp_anneal = use_temp_anneal
        self.min_temp = min_temp

    def on_epoch_start(self):
        if self.use_temp_anneal:
            self.temp = (1 - self.trainer.current_epoch / self.trainer.max_epochs) * (self.init_temp - self.min_temp) + self.min_temp
            self.temp = max(self.temp, self.min_temp)

            for _, nas_module in self.nas_modules:
                nas_module.temp = self.temp

        return self.model.on_epoch_start()

    @staticmethod
    def match_and_replace():
        inputchoice_replace = get_naive_match_and_replace(InputChoice, SNASInputChoice)
        layerchoice_replace = get_naive_match_and_replace(LayerChoice, SNASLayerChoice)

        return [inputchoice_replace, layerchoice_replace]
