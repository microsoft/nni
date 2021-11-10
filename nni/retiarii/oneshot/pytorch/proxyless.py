# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interface import BaseOneShotTrainer
from .utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device


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
        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
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
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Input choice is not supported for ProxylessNAS.')


class HardwareLatencyEstimator:
    def __init__(self, applied_hardware, model, dummy_input=(1, 3, 224, 224), dump_lat_table='data/latency_table.yaml'):
        import nn_meter  # pylint: disable=import-error
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
        self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)
        self.block_latency_table = self._form_latency_table(model, dummy_input, dump_lat_table=dump_lat_table)

    def _form_latency_table(self, model, dummy_input, dump_lat_table):
        latency_table = {}

        from nni.retiarii.converter import convert_to_graph
        from nni.retiarii.converter.graph_gen import GraphConverterWithShape
        from nni.retiarii.converter.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node
        script_module = torch.jit.script(model)
        base_model_ir = convert_to_graph(script_module, model,
                                         converter=GraphConverterWithShape(), dummy_input=torch.randn(*dummy_input))

        # form the latency of layerchoice blocks for the latency table
        temp_ir_model = base_model_ir.fork()
        cell_nodes = base_model_ir.get_cell_nodes()
        layerchoice_nodes = [node for node in cell_nodes if is_layerchoice_node(node)]
        for lc_node in layerchoice_nodes:
            cand_lat = {}
            for candidate in lc_node.operation.parameters['candidates']:
                node_graph = base_model_ir.graphs.get(candidate)
                if node_graph is not None:
                    temp_ir_model._root_graph_name = node_graph.name
                    latency = self.latency_predictor.predict(temp_ir_model, model_type = 'nni-ir')
                else:
                    _logger.warning(f"Could not found graph for layerchoice candidate {candidate}")
                    latency = 0
                cand_lat[candidate.split('_')[-1]] = float(latency)
            latency_table[lc_node.operation.parameters['label']] = cand_lat

        # form the latency of the stationary block in the latency table
        temp_ir_model._root_graph_name = base_model_ir._root_graph_name
        temp_ir_model = flatten_model_graph_without_layerchoice(temp_ir_model)
        latency = self.latency_predictor.predict(temp_ir_model, model_type = 'nni-ir')
        latency_table['stationary_block'] = {'root': float(latency)}

        # save latency table
        if dump_lat_table:
            import os, yaml
            os.makedirs(os.path.dirname(dump_lat_table), exist_ok=True)
            with open(dump_lat_table, 'a') as fp:
                yaml.dump([{
                    "applied_hardware": self.predictor_name,
                    'latency_table': latency_table
                    }], fp)
        _logger.info("Latency lookup table form done")

        return latency_table

    def cal_expected_latency(self, current_architecture_prob):
        lat = self.block_latency_table['stationary_block']['root']
        for module_name, probs in current_architecture_prob.items():
            assert len(probs) == len(self.block_latency_table[module_name])
            lat += torch.sum(torch.tensor([probs[i] * self.block_latency_table[module_name][str(i)]
                                for i in range(len(probs))]))
        return lat

    def export_latency(self, current_architecture):
        lat = self.block_latency_table['stationary_block']['root']
        for module_name, selected_module in current_architecture.items():
            lat += self.block_latency_table[module_name][str(selected_module)]
        return lat


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
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    warmup_epochs : int
        Number of epochs to warmup model parameters.
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
    grad_reg_loss_type: string
        Regularization type to add hardware related loss, allowed types include
        - ``"mul#log"``: ``regularized_loss = (torch.log(expected_latency) / math.log(self.ref_latency)) ** beta``
        - ``"add#linear"``: ``regularized_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency``
        - None: do not apply loss regularization.
    grad_reg_loss_params: dict
        Regularization params, allowed params include
        - ``"alpha"`` and ``"beta"`` is required when ``grad_reg_loss_type == "mul#log"``
        - ``"lambda"`` is required when ``grad_reg_loss_type == "add#linear"``
    applied_hardware: string
        Applied hardware for to constraint the model's latency. Latency is predicted by Microsoft
        nn-Meter (https://github.com/microsoft/nn-Meter).
    dummy_input: tuple
        The dummy input shape when applied to the target hardware.
    ref_latency: float
        Reference latency value in the applied hardware (ms).
    """

    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, warmup_epochs=0,
                 batch_size=64, workers=4, device=None, log_frequency=None,
                 arc_learning_rate=1.0E-3,
                 grad_reg_loss_type=None, grad_reg_loss_params=None,
                 applied_hardware=None, dummy_input=(1, 3, 224, 224),
                 ref_latency=65.0):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        # latency predictor
        if applied_hardware:
            self.latency_estimator = HardwareLatencyEstimator(applied_hardware, self.model, dummy_input)
        else:
            self.latency_estimator = None
        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        self.ref_latency = ref_latency

        self.model.to(self.device)
        self.nas_modules = []
        replace_layer_choice(self.model, ProxylessLayerChoice, self.nas_modules)
        replace_input_choice(self.model, ProxylessInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.optimizer = optimizer
        # we do not support deduplicate control parameters with same label (like DARTS) yet.
        self.ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], arc_learning_rate,
                                           weight_decay=0, betas=(0, 0.999), eps=1e-8)
        self._init_dataloader()

    def _init_dataloader(self):
        n_train = len(self.dataset)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers)

    def _train_one_epoch(self, epoch):
        self.model.train()
        meters = AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
            val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

            if epoch >= self.warmup_epochs:
                # 1) train architecture parameters
                for _, module in self.nas_modules:
                    module.resample()
                self.ctrl_optim.zero_grad()
                logits, loss = self._logits_and_loss_for_arch_update(val_X, val_y)
                loss.backward()
                for _, module in self.nas_modules:
                    module.finalize_grad()
                self.ctrl_optim.step()

            # 2) train model parameters
            for _, module in self.nas_modules:
                module.resample()
            self.optimizer.zero_grad()
            logits, loss = self._logits_and_loss_for_weight_update(trn_X, trn_y)
            loss.backward()
            self.optimizer.step()
            metrics = self.metrics(logits, trn_y)
            metrics["loss"] = loss.item()
            if self.latency_estimator:
                metrics["latency"] = self._export_latency()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                _logger.info("Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1,
                             self.num_epochs, step + 1, len(self.train_loader), meters)

    def _logits_and_loss_for_arch_update(self, X, y):
        ''' return logits and loss for architecture parameter update '''
        logits = self.model(X)
        ce_loss = self.loss(logits, y)
        if not self.latency_estimator:
            return logits, ce_loss

        current_architecture_prob = {}
        for module_name, module in self.nas_modules:
            probs = module.export_prob()
            current_architecture_prob[module_name] = probs
        expected_latency = self.latency_estimator.cal_expected_latency(current_architecture_prob)

        if self.reg_loss_type == 'mul#log':
            import math
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_latency) / math.log(self.ref_latency)) ** beta
            return logits, alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency
            return logits, ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return logits, ce_loss
        else:
            raise ValueError(f'Do not support: {self.reg_loss_type}')

    def _logits_and_loss_for_weight_update(self, X, y):
        ''' return logits and loss for weight parameter update '''
        logits = self.model(X)
        loss = self.loss(logits, y)
        return logits, loss

    def _export_latency(self):
        current_architecture = {}
        for module_name, module in self.nas_modules:
            selected_module = module.export()
            current_architecture[module_name] = selected_module
        return self.latency_estimator.export_latency(current_architecture)

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)

    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
