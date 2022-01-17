# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from typing import List, Dict, Tuple, Callable, Optional

from schema import And, Or, Optional as SchemaOptional, SchemaError
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from nni.common.serializer import Traceable
from nni.algorithms.compression.v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression.v2.pytorch.utils import CompressorSchema, config_list_canonical, OptimizerConstructHelper

from .tools import (
    DataCollector,
    HookCollectorInfo,
    WeightDataCollector,
    WeightTrainerBasedDataCollector,
    SingleHookTrainerBasedDataCollector
)

from .tools import (
    MetricsCalculator,
    NormMetricsCalculator,
    MultiDataNormMetricsCalculator,
    DistMetricsCalculator,
    APoZRankMetricsCalculator,
    MeanRankMetricsCalculator
)

from .tools import (
    SparsityAllocator,
    NormalSparsityAllocator,
    GlobalSparsityAllocator,
    Conv2dDependencyAwareAllocator
)

_logger = logging.getLogger(__name__)

__all__ = ['LevelPruner', 'L1NormPruner', 'L2NormPruner', 'FPGMPruner', 'SlimPruner', 'ActivationPruner',
           'ActivationAPoZRankPruner', 'ActivationMeanRankPruner', 'TaylorFOWeightPruner', 'ADMMPruner']

NORMAL_SCHEMA = {
    Or('sparsity', 'sparsity_per_layer'): And(float, lambda n: 0 <= n < 1),
    SchemaOptional('op_types'): [str],
    SchemaOptional('op_names'): [str],
    SchemaOptional('op_partial_names'): [str]
}

GLOBAL_SCHEMA = {
    'total_sparsity': And(float, lambda n: 0 <= n < 1),
    SchemaOptional('max_sparsity_per_layer'): And(float, lambda n: 0 < n <= 1),
    SchemaOptional('op_types'): [str],
    SchemaOptional('op_names'): [str],
    SchemaOptional('op_partial_names'): [str]
}

EXCLUDE_SCHEMA = {
    'exclude': bool,
    SchemaOptional('op_types'): [str],
    SchemaOptional('op_names'): [str],
    SchemaOptional('op_partial_names'): [str]
}

INTERNAL_SCHEMA = {
    'total_sparsity': And(float, lambda n: 0 <= n < 1),
    SchemaOptional('max_sparsity_per_layer'): {str: float},
    SchemaOptional('op_types'): [str],
    SchemaOptional('op_names'): [str]
}


class BasicPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict]):
        self.data_collector: DataCollector = None
        self.metrics_calculator: MetricsCalculator = None
        self.sparsity_allocator: SparsityAllocator = None

        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        self._validate_config_before_canonical(model, config_list)
        self.config_list = config_list_canonical(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        pass

    def reset(self, model: Optional[Module], config_list: Optional[List[Dict]]):
        super().reset(model=model, config_list=config_list)
        self.reset_tools()

    def reset_tools(self):
        """
        This function is used to reset `self.data_collector`, `self.metrics_calculator` and `self.sparsity_allocator`.
        The subclass needs to implement this function to complete the pruning process.
        See `compress()` to understand how NNI use these three part to generate mask for the bound model.
        """
        raise NotImplementedError()

    def compress(self) -> Tuple[Module, Dict]:
        """
        Used to generate the mask. Pruning process is divided in three stages.
        `self.data_collector` collect the data used to calculate the specify metric.
        `self.metrics_calculator` calculate the metric and `self.sparsity_allocator` generate the mask depend on the metric.

        Returns
        -------
        Tuple[Module, Dict]
            Return the wrapped model and mask.
        """
        data = self.data_collector.collect()
        _logger.debug('Collected Data:\n%s', data)
        metrics = self.metrics_calculator.calculate_metrics(data)
        _logger.debug('Metrics Calculate:\n%s', metrics)
        masks = self.sparsity_allocator.generate_sparsity(metrics)
        _logger.debug('Masks:\n%s', masks)
        self.load_masks(masks)
        return self.bound_model, masks


class LevelPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Operation types to be pruned.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    """

    def __init__(self, model: Module, config_list: List[Dict]):
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self)


class NormPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    p : int
        The order of norm.
    mode : str
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict], p: int,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.p = p
        self.mode = mode
        self.dummy_input = dummy_input
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)

        schema.validate(config_list)

    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator(p=self.p, dim=0)
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = Conv2dDependencyAwareAllocator(self, 0, self.dummy_input)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')


class L1NormPruner(NormPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in L1NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode : str
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the l1-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        super().__init__(model, config_list, 1, mode, dummy_input)


class L2NormPruner(NormPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in L1NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode : str
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the l2-norm of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        super().__init__(model, config_list, 2, mode, dummy_input)


class FPGMPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in FPGMPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode : str
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the FPGM of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        self.dummy_input = dummy_input
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)

        schema.validate(config_list)

    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = DistMetricsCalculator(p=2, dim=0)
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = Conv2dDependencyAwareAllocator(self, 0, self.dummy_input)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')


class SlimPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - total_sparsity : This is to specify the total sparsity for all layers in this config, each layer may have different sparsity.
            - max_sparsity_per_layer : Always used with total_sparsity. Limit the max sparsity of each layer.
            - op_types : Only BatchNorm2d is supported in SlimPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    trainer : Callable[[Module, Optimizer, Callable], None]
        A callable function used to train model or just inference. Take model, optimizer, criterion as input.
        The model will be trained or inferenced `training_epochs` epochs.

        Example::

            def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                training = model.training
                model.train(mode=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                    optimizer.step()
                model.train(mode=training)
    traced_optimizer : nni.common.serializer.Traceable(torch.optim.Optimizer)
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters()).
    criterion : Callable[[Tensor, Tensor], Tensor]
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    training_epochs : int
        The epoch number for training model to sparsify the BN weight.
    scale : float
        Penalty parameter for sparsification, which could reduce overfitting.
    mode : str
        'normal' or 'global'.
        If prune the model in a global way, all layer weights with same config will be considered uniformly.
        That means a single layer may not reach or exceed the sparsity setting in config,
        but the total pruned weights meet the sparsity setting.
    """

    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor],
                 training_epochs: int, scale: float = 0.0001, mode='global'):
        self.mode = mode
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.training_epochs = training_epochs
        self._scale = scale
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        if self.mode == 'global':
            schema_list.append(deepcopy(GLOBAL_SCHEMA))
        else:
            schema_list.append(deepcopy(NORMAL_SCHEMA))
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['BatchNorm2d']
        schema = CompressorSchema(schema_list, model, _logger)

        try:
            schema.validate(config_list)
        except SchemaError as e:
            if "Missing key: 'total_sparsity'" in str(e):
                _logger.error('`config_list` validation failed. If global mode is set in this pruner, `sparsity_per_layer` and `sparsity` are not supported, make sure `total_sparsity` is set in config_list.')
            raise e

    def criterion_patch(self, criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def patched_criterion(input_tensor: Tensor, target: Tensor):
            sum_l1 = 0
            for _, wrapper in self.get_modules_wrapper().items():
                sum_l1 += torch.norm(wrapper.module.weight, p=1)
            return criterion(input_tensor, target) + self._scale * sum_l1
        return patched_criterion

    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                  self.training_epochs, criterion_patch=self.criterion_patch)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self)
            else:
                raise NotImplementedError('Only support mode `normal` and `global`')


class ActivationPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in ActivationPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    trainer : Callable[[Module, Optimizer, Callable], None]
        A callable function used to train model or just inference. Take model, optimizer, criterion as input.
        The model will be trained or inferenced `training_epochs` epochs.

        Example::

            def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                training = model.training
                model.train(mode=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                    optimizer.step()
                model.train(mode=training)
    traced_optimizer : nni.common.serializer.Traceable(torch.optim.Optimizer)
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters()).
    criterion : Callable[[Tensor, Tensor], Tensor]
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    training_batches
        The batch number used to collect activations.
    mode : str
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the activation-based metrics and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor], training_batches: int, activation: str = 'relu',
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        self.dummy_input = dummy_input
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.training_batches = training_batches
        self._activation = self._choose_activation(activation)
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)

        schema.validate(config_list)

    def _choose_activation(self, activation: str = 'relu') -> Callable:
        if activation == 'relu':
            return nn.functional.relu
        elif activation == 'relu6':
            return nn.functional.relu6
        else:
            raise 'Unsupported activatoin {}'.format(activation)

    def _collector(self, buffer: List) -> Callable[[Module, Tensor, Tensor], None]:
        assert len(buffer) == 0, 'Buffer pass to activation pruner collector is not empty.'
        # The length of the buffer used in this pruner will always be 2.
        # buffer[0] is the number of how many batches are counted in buffer[1].
        # buffer[1] is a tensor and the size of buffer[1] is same as the activation.
        buffer.append(0)

        def collect_activation(_module: Module, _input: Tensor, output: Tensor):
            if len(buffer) == 1:
                buffer.append(torch.zeros_like(output))
            if buffer[0] < self.training_batches:
                buffer[1] += self._activation_trans(output)
                buffer[0] += 1
        return collect_activation

    def _activation_trans(self, output: Tensor) -> Tensor:
        raise NotImplementedError()

    def reset_tools(self):
        collector_info = HookCollectorInfo([layer_info for layer_info, _ in self._detect_modules_to_compress()], 'forward', self._collector)
        if self.data_collector is None:
            self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                      1, collector_infos=[collector_info])
        else:
            self.data_collector.reset(collector_infos=[collector_info])
        if self.metrics_calculator is None:
            self.metrics_calculator = self._get_metrics_calculator()
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = Conv2dDependencyAwareAllocator(self, 0, self.dummy_input)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')

    def _get_metrics_calculator(self) -> MetricsCalculator:
        raise NotImplementedError()


class ActivationAPoZRankPruner(ActivationPruner):
    def _activation_trans(self, output: Tensor) -> Tensor:
        # return a matrix that the position of zero in `output` is one, others is zero.
        return torch.eq(self._activation(output.detach()), torch.zeros_like(output)).type_as(output)

    def _get_metrics_calculator(self) -> MetricsCalculator:
        return APoZRankMetricsCalculator(dim=1)


class ActivationMeanRankPruner(ActivationPruner):
    def _activation_trans(self, output: Tensor) -> Tensor:
        # return the activation of `output` directly.
        return self._activation(output.detach())

    def _get_metrics_calculator(self) -> MetricsCalculator:
        return MeanRankMetricsCalculator(dim=1)


class TaylorFOWeightPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - total_sparsity : This is to specify the total sparsity for all layers in this config, each layer may have different sparsity.
            - max_sparsity_per_layer : Always used with total_sparsity. Limit the max sparsity of each layer.
            - op_types : Conv2d and Linear are supported in TaylorFOWeightPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    trainer : Callable[[Module, Optimizer, Callable]
        A callable function used to train model or just inference. Take model, optimizer, criterion as input.
        The model will be trained or inferenced `training_epochs` epochs.

        Example::

            def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                training = model.training
                model.train(mode=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                    optimizer.step()
                model.train(mode=training)
    traced_optimizer : nni.common.serializer.Traceable(torch.optim.Optimizer)
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters()).
    criterion : Callable[[Tensor, Tensor], Tensor]
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    training_batches : int
        The batch number used to collect activations.
    mode : str
        'normal', 'dependency_aware' or 'global'.

        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the taylorFO and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.

        If prune the model in a global way, all layer weights with same config will be considered uniformly.
        That means a single layer may not reach or exceed the sparsity setting in config,
        but the total pruned weights meet the sparsity setting.
    dummy_input : Optional[torch.Tensor]
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.
    """

    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor], training_batches: int,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        self.mode = mode
        self.dummy_input = dummy_input
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.training_batches = training_batches
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        if self.mode == 'global':
            schema_list.append(deepcopy(GLOBAL_SCHEMA))
        else:
            schema_list.append(deepcopy(NORMAL_SCHEMA))
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)

        try:
            schema.validate(config_list)
        except SchemaError as e:
            if "Missing key: 'total_sparsity'" in str(e):
                _logger.error('`config_list` validation failed. If global mode is set in this pruner, `sparsity_per_layer` and `sparsity` are not supported, make sure `total_sparsity` is set in config_list.')
            raise e

    def _collector(self, buffer: List, weight_tensor: Tensor) -> Callable[[Tensor], None]:
        assert len(buffer) == 0, 'Buffer pass to taylor pruner collector is not empty.'
        buffer.append(0)
        buffer.append(torch.zeros_like(weight_tensor))

        def collect_taylor(grad: Tensor):
            if buffer[0] < self.training_batches:
                buffer[1] += self._calculate_taylor_expansion(weight_tensor, grad)
                buffer[0] += 1
        return collect_taylor

    def _calculate_taylor_expansion(self, weight_tensor: Tensor, grad: Tensor) -> Tensor:
        return (weight_tensor.detach() * grad.detach()).data.pow(2)

    def reset_tools(self):
        hook_targets = {layer_info.name: layer_info.module.weight for layer_info, _ in self._detect_modules_to_compress()}
        collector_info = HookCollectorInfo(hook_targets, 'tensor', self._collector)
        if self.data_collector is None:
            self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                      1, collector_infos=[collector_info])
        else:
            self.data_collector.reset(collector_infos=[collector_info])
        if self.metrics_calculator is None:
            self.metrics_calculator = MultiDataNormMetricsCalculator(p=1, dim=0)
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self, dim=0)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = Conv2dDependencyAwareAllocator(self, 0, self.dummy_input)
            else:
                raise NotImplementedError('Only support mode `normal`, `global` and `dependency_aware`')


class ADMMPruner(BasicPruner):
    """
    ADMM (Alternating Direction Method of Multipliers) Pruner is a kind of mathematical optimization technique.
    The metric used in this pruner is the absolute value of the weight.
    In each iteration, the weight with small magnitudes will be set to zero.
    Only in the final iteration, the mask will be generated and apply to model wrapper.

    The original paper refer to: https://arxiv.org/abs/1804.03294.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - rho : Penalty parameters in ADMM algorithm.
            - op_types : Operation types to be pruned.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    trainer : Callable[[Module, Optimizer, Callable]
        A callable function used to train model or just inference. Take model, optimizer, criterion as input.
        The model will be trained or inferenced `training_epochs` epochs.

        Example::

            def trainer(model: Module, optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]):
                training = model.training
                model.train(mode=True)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    # If you don't want to update the model, you can skip `optimizer.step()`, and set train mode False.
                    optimizer.step()
                model.train(mode=training)
    traced_optimizer : nni.common.serializer.Traceable(torch.optim.Optimizer)
        The traced optimizer instance which the optimizer class is wrapped by nni.trace.
        E.g. traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters()).
    criterion : Callable[[Tensor, Tensor], Tensor]
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    iterations : int
        The total iteration number in admm pruning algorithm.
    training_epochs : int
        The epoch number for training model in each iteration.
    """

    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor], iterations: int, training_epochs: int):
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.iterations = iterations
        self.training_epochs = training_epochs
        super().__init__(model, config_list)

    def reset(self, model: Optional[Module], config_list: Optional[List[Dict]]):
        super().reset(model, config_list)
        self.Z = {name: wrapper.module.weight.data.clone().detach() for name, wrapper in self.get_modules_wrapper().items()}
        self.U = {name: torch.zeros_like(z).to(z.device) for name, z in self.Z.items()}

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for schema in schema_list:
            schema.update({SchemaOptional('rho'): And(float, lambda n: n > 0)})
        schema_list.append(deepcopy(EXCLUDE_SCHEMA))
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    def criterion_patch(self, origin_criterion: Callable[[Tensor, Tensor], Tensor]):
        def patched_criterion(output: Tensor, target: Tensor):
            penalty = torch.tensor(0.0).to(output.device)
            for name, wrapper in self.get_modules_wrapper().items():
                rho = wrapper.config.get('rho', 1e-4)
                penalty += (rho / 2) * torch.sqrt(torch.norm(wrapper.module.weight - self.Z[name] + self.U[name]))
            return origin_criterion(output, target) + penalty
        return patched_criterion

    def reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                  self.training_epochs, criterion_patch=self.criterion_patch)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self)

    def compress(self) -> Tuple[Module, Dict]:
        """
        Returns
        -------
        Tuple[Module, Dict]
            Return the wrapped model and mask.
        """
        for i in range(self.iterations):
            _logger.info('======= ADMM Iteration %d Start =======', i)
            data = self.data_collector.collect()

            for name, weight in data.items():
                self.Z[name] = weight + self.U[name]
            metrics = self.metrics_calculator.calculate_metrics(self.Z)
            masks = self.sparsity_allocator.generate_sparsity(metrics)

            for name, mask in masks.items():
                self.Z[name] = self.Z[name].mul(mask['weight'])
                self.U[name] = self.U[name] + data[name] - self.Z[name]

        self.Z = None
        self.U = None
        torch.cuda.empty_cache()

        metrics = self.metrics_calculator.calculate_metrics(data)
        masks = self.sparsity_allocator.generate_sparsity(metrics)

        self.load_masks(masks)
        return self.bound_model, masks
