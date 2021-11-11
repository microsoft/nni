# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from schema import Optional as SchemaOptional
from typing import Dict, List, Tuple, Callable

import torch
from torch import autograd, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer, SGD

from nni.algorithms.compression.v2.pytorch.base.compressor import Compressor, _setattr, LayerInfo
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import BasicPruner, NORMAL_SCHEMA, EXCLUDE_SCHEMA, INTERNAL_SCHEMA
from nni.algorithms.compression.v2.pytorch.utils import CompressorSchema

from .tools.base import TrainerBasedDataCollector

from .tools import (
    NormMetricsCalculator,
    NormalSparsityAllocator
)

_logger = logging.getLogger(__name__)


class PrunerScoredModuleWrapper(Module):
    def __init__(self, module: Module, module_name: str, config: Dict, pruner: Compressor):
        """
        Wrap an module to enable data parallel, forward method customization and buffer registeration.
        Different from `PrunerModuleWrapper`, `PrunerScoredModuleWrapper` will record the gradient.

        Parameters
        ----------
        module
            The module user wants to compress.
        config
            The configurations that users specify for compression.
        module_name
            The name of the module to compress, wrapper module shares same name.
        pruner
            The pruner used to calculate mask.
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        # config and pruner
        self.config = config
        self.pruner = pruner

        self.weight = Parameter(torch.empty(self.module.weight.size()))
        self.weight.data = self.module.weight.data

        self.weight_score = Parameter(torch.empty(self.weight.size()))
        torch.nn.init.constant_(self.weight_score, val=0.0)

        # register buffer for mask
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
            self.bias = Parameter(torch.empty(self.module.bias.size()))
            self.bias.data = self.module.bias.data
        else:
            self.register_buffer("bias_mask", None)

    def _weight2buffer(self):
        delattr(self.module, 'weight')
        self.module.register_buffer('weight', self.weight.data)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            delattr(self.module, 'bias')
            self.module.register_buffer('bias', self.bias.data)

    def _weight2parameter(self):
        delattr(self.module, 'weight')
        self.module.weight = Parameter(torch.empty(self.weight.size()))
        self.module.weight.data = torch.mul(self.weight, self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            delattr(self.module, 'bias')
            self.module.bias = Parameter(torch.empty(self.bias.size()))
            self.module.bias.data = torch.mul(self.bias, self.bias_mask)

    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight = torch.mul(self.weight, _StraightThrough.apply(self.weight_score, self.weight_mask))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias = torch.mul(self.bias, self.bias_mask)
        return self.module(*inputs)


class _StraightThrough(autograd.Function):
    @staticmethod
    def forward(self, score, masks):
        return masks

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class WeightScoreTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Collect all wrapper weight score.
    """
    def _reset_optimizer(self):
        if self._origin_optimizer_cls is not None:
            optimizer_grouped_parameters = [{
                "params": [p for n, p in self.compressor.bound_model.named_parameters() if "weight_score" not in n and p.requires_grad]
            }]
            if self._origin_optimizer_cls.__name__ == 'SGD':
                self.optimizer = self._origin_optimizer_cls(optimizer_grouped_parameters, lr=0.001)
            else:
                self.optimizer = self._origin_optimizer_cls(optimizer_grouped_parameters)
            self.optimizer.load_state_dict(self._origin_optimizer_state_dict)
        else:
            self.optimizer = None

    def collect(self) -> Dict[str, Tensor]:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = wrapper.weight_score.data.clone().detach()
        return data


class MovementPruner(BasicPruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int, warm_up_step: int,
                 cool_down_beginning_step: int):
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_epochs = training_epochs
        self.warm_up_step = warm_up_step
        self.cool_down_beginning_step = cool_down_beginning_step
        assert self.warm_up_step < self.cool_down_beginning_step
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    def cubic_schedule(self, current_step: int):
        if self.warm_up_step < current_step <= self.cool_down_beginning_step:
            wrapper_dict = self.get_modules_wrapper()
            for config in self.config_list:
                config = config.copy()
                total_sparsity = config['total_sparsity'] * (1 - (1 - (current_step - self.warm_up_step) / (self.cool_down_beginning_step - self.warm_up_step)) ** 3)
                for op_name in config['op_names']:
                    wrapper_dict[op_name].config['total_sparsity'] = total_sparsity

    def reset_tools(self):
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self)

        # use a SGD to update the weight_score
        params = [{"params": [p for n, p in self.bound_model.named_parameters() if "weight_score" in n and p.requires_grad]}]
        optimizer = SGD(params, 0.001)
        self.step_counter = 0

        # update the masks after each optimzier step
        def _optimizer_patch():
            optimizer.step()
            optimizer.zero_grad()
            self.step_counter += 1
            if self.step_counter > self.warm_up_step:
                self.cubic_schedule(self.step_counter)
                data = {}
                for _, wrapper in self.get_modules_wrapper().items():
                    data[wrapper.name] = wrapper.weight_score.data
                metrics = self.metrics_calculator.calculate_metrics(data)
                masks = self.sparsity_allocator.generate_sparsity(metrics)
                self.load_masks(masks)

        if self.data_collector is None:
            self.data_collector = WeightScoreTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion, self.training_epochs, opt_after_tasks=[_optimizer_patch])
        else:
            self.data_collector.reset()

    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        """
        if not self.is_wrapped:
            for _, wrapper in reversed(self.get_modules_wrapper().items()):
                _setattr(self.bound_model, wrapper.name, wrapper)
                wrapper._weight2buffer()
            self.is_wrapped = True

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        """
        if self.is_wrapped:
            for _, wrapper in self.get_modules_wrapper().items():
                _setattr(self.bound_model, wrapper.name, wrapper.module)
                wrapper._weight2parameter()
            self.is_wrapped = False

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerScoredModuleWrapper(layer.module, layer.name, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def compress(self) -> Tuple[Module, Dict]:
        # sparsity grow from 0
        for _, wrapper in self.get_modules_wrapper().items():
            wrapper.config['total_sparsity'] = 0
        return super().compress()
