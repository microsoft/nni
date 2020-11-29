# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.nas.pytorch.mutables import LayerChoice

from ..interface import BaseOneShotTrainer
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


_logger = logging.getLogger(__name__)


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


class ProxylessLayerChoice(nn.Module):
    def __init__(self, ops):
        super(ProxylessLayerChoice, self).__init__()
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



class ProxylessInputChoice(nn.Module):
    ...


class ProxylessTrainer(BaseOneShotTrainer):
    """
    Proxyless trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    num_epochs : int
        Number of epochs planned for training.
    dataset_train : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    dataset_valid : Dataset
        Dataset for testing.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """
    def __init__(self, model, loss, metrics,
                 num_epochs, dataset_train, dataset_valid,
                 learning_rate=2.5E-3, batch_size=64, workers=4, device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, unrolled=False):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        self.proxyless_modules = replace_layer_choice(self.model, ProxylessLayerChoice) + replace_input_choice(self.mode, ProxylessInputChoice)

        self.model_optim = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.ctrl_optim = torch.optim.Adam([c.alpha for c in self.proxyless_modules], arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled

        n_train = len(self.dataset_train)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=workers)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=batch_size,
                                                       num_workers=workers)

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = trn_X.to(self.device), trn_y.to(self.device)
            val_X, val_y = val_X.to(self.device), val_y.to(self.device)

            # 1) train architecture parameters
            for module in self.proxyless_modules:
                module.resample()
            self.ctrl_optim.zero_grad()
            logits, loss = self._logits_and_loss(val_X, val_y)
            loss.backward()
            for module in self.proxyless_modules:
                module.finalize_grad()
            self.ctrl_optim.step()

            # 2) train model parameters
            for module in self.proxyless_modules:
                module.resample()
            self.model_optim.zero_grad()
            logits, loss = self._logits_and_loss(trn_X, trn_y)
            loss.backward()
            self.model_optim.step()

            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters)

    def _logits_and_loss(self, X, y):
        logits = self.model(X)
        loss = self.loss(logits, y)
        return logits, loss

    def fit(self):
        for i in range(self.epochs):
            self.train_one_epoch(i)

    def export(self):
        # TODO
        ...
