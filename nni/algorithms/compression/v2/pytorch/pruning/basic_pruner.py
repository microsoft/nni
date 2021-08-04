# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Dict, Tuple, Callable, Optional

from schema import And, Optional as SchemaOptional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer

from nni.algorithms.compression.v2.pytorch.base.pruner import Pruner
from nni.algorithms.compression.v2.pytorch.utils.config_validation import PrunerSchema

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

__all__ = ['LevelPruner', 'L1FilterPruner', 'L2FilterPruner', 'FPGMPruner', 'SlimPruner', 'ActivationFilterPruner',
           'ActivationAPoZRankFilterPruner', 'ActivationMeanRankFilterPruner', 'TaylorFOWeightFilterPruner']


class OneShotPruner(Pruner):
    def __init__(self, model: Module, config_list: List[Dict]):
        self.data_collector: DataCollector = None
        self.metrics_calculator: MetricsCalculator = None
        self.sparsity_allocator: SparsityAllocator = None
        super().__init__(model, config_list)

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


class LevelPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict]):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Operation types to prune.
        """
        self.mode = 'normal'
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            SchemaOptional('op_types'): [str],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightDataCollector(self)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self)


class NormFilterPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict], p: int,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in NormFilterPruner.
        p
            The order of norm.
        mode
            'normal' or 'dependency_aware'.
            If prune the model in a dependency-aware way, this pruner will
            prune the model according to the norm of weights and the channel-dependency or
            group-dependency of the model. In this way, the pruner will force the conv layers
            that have dependencies to prune the same channels, so the speedup module can better
            harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
            , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
            dependency between the conv layers.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        self.p = p
        self.mode = mode
        self.dummy_input = dummy_input
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _reset_tools(self):
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


class L1FilterPruner(NormFilterPruner):
    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in L1FilterPruner.
        mode
            'normal' or 'dependency_aware'.
            If prune the model in a dependency-aware way, this pruner will
            prune the model according to the l1-norm of weights and the channel-dependency or
            group-dependency of the model. In this way, the pruner will force the conv layers
            that have dependencies to prune the same channels, so the speedup module can better
            harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
            , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
            dependency between the conv layers.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        super().__init__(model, config_list, 1, mode, dummy_input)


class L2FilterPruner(NormFilterPruner):
    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in L2FilterPruner.
        mode
            'normal' or 'dependency_aware'.
            If prune the model in a dependency-aware way, this pruner will
            prune the model according to the l2-norm of weights and the channel-dependency or
            group-dependency of the model. In this way, the pruner will force the conv layers
            that have dependencies to prune the same channels, so the speedup module can better
            harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
            , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
            dependency between the conv layers.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        super().__init__(model, config_list, 2, mode, dummy_input)


class FPGMPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in FPGMPruner.
        mode
            'normal' or 'dependency_aware'.
            If prune the model in a dependency-aware way, this pruner will
            prune the model according to the l1-norm of weights and the channel-dependency or
            group-dependency of the model. In this way, the pruner will force the conv layers
            that have dependencies to prune the same channels, so the speedup module can better
            harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
            , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
            dependency between the conv layers.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        self.mode = mode
        self.dummy_input = dummy_input
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _reset_tools(self):
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


class SlimPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor],
                 training_epochs: int, scale: float = 0.0001, mode='global', max_sparsity_per_layer: float = 1):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only BatchNorm2D is supported in SlimPruner.
        trainer
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
        optimizer
            The optimizer instance used in trainer. Note that this optimizer might be patched during collect data,
            so do not use this optimizer in other places.
        criterion
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        training_epochs
            The epoch number for training model to sparsify the BN weight.
        mode
            'normal' or 'global'.
            If prune the model in a global way, all layer weights with same config will be considered uniformly.
            That means a single layer may not reach or exceed the sparsity setting in config,
            but the total pruned weights meet the sparsity setting.
        max_sparsity_per_layer
            The max sparsity per layer constraint, to prevent all weight in a layer be pruned in global mode.
        """
        self.mode = mode
        assert 0 < max_sparsity_per_layer <= 1, 'max_sparsity_per_layer must in range (0, 1].'
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_epochs = training_epochs
        self._scale = scale
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            'op_types': ['BatchNorm2d'],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def criterion_patch(self, criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def patched_criterion(input_tensor: Tensor, target: Tensor):
            sum_l1 = 0
            for _, wrapper in self.get_modules_wrapper().items():
                sum_l1 += torch.norm(wrapper.module.weight.data, p=1)
            return criterion(input_tensor, target) + self._scale * sum_l1
        return patched_criterion

    def _reset_tools(self):
        if self.data_collector is None:
            self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion,
                                                                  self.training_epochs, criterion_patch=self.criterion_patch)
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = NormMetricsCalculator()
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self, 0, self.max_sparsity_per_layer)
            else:
                raise NotImplementedError('Only support mode `normal` and `global`')


class ActivationFilterPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor], training_batches: int, activation: str = 'relu',
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in ActivationFilterPruner.
        trainer
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
        optimizer
            The optimizer instance used in trainer. Note that this optimizer might be patched during collect data,
            so do not use this optimizer in other places.
        criterion
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        training_batches
            The batch number used to collect activations.
        mode
            'normal' or 'dependency_aware'.
            If prune the model in a dependency-aware way, this pruner will
            prune the model according to the activation-based metrics and the channel-dependency or
            group-dependency of the model. In this way, the pruner will force the conv layers
            that have dependencies to prune the same channels, so the speedup module can better
            harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
            , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
            dependency between the conv layers.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        self.mode = mode
        self.dummy_input = dummy_input
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_batches = training_batches
        self._activation = self._choose_activation(activation)
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _choose_activation(self, activation: str = 'relu') -> Callable:
        if activation == 'relu':
            return nn.functional.relu
        elif activation == 'relu6':
            return nn.functional.relu6
        else:
            raise 'Unsupported activatoin {}'.format(activation)

    def _collector(self, buffer: List) -> Callable[[Module, Tensor, Tensor], None]:
        def collect_activation(_module: Module, _input: Tensor, output: Tensor):
            if len(buffer) < self.training_batches:
                buffer.append(self._activation(output.detach()))
        return collect_activation

    def _reset_tools(self):
        collector_info = HookCollectorInfo([layer_info for layer_info, _ in self._detect_modules_to_compress()], 'forward', self._collector)
        if self.data_collector is None:
            self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion,
                                                                      1, collector_infos=[collector_info])
        else:
            self.data_collector.reset()
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


class ActivationAPoZRankFilterPruner(ActivationFilterPruner):
    def _get_metrics_calculator(self) -> MetricsCalculator:
        return APoZRankMetricsCalculator(dim=1)


class ActivationMeanRankFilterPruner(ActivationFilterPruner):
    def _get_metrics_calculator(self) -> MetricsCalculator:
        return MeanRankMetricsCalculator(dim=1)


class TaylorFOWeightFilterPruner(OneShotPruner):
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor], training_batches: int,
                 mode: str = 'normal', max_sparsity_per_layer: float = 1, dummy_input: Optional[Tensor] = None):
        """
        Parameters
        ----------
        model
            Model to be pruned
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Only Conv2d is supported in TaylorFOWeightFilterPruner.
        trainer
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
        optimizer
            The optimizer instance used in trainer. Note that this optimizer might be patched during collect data,
            so do not use this optimizer in other places.
        criterion
            The criterion function used in trainer. Take model output and target value as input, and return the loss.
        training_batches
            The batch number used to collect activations.
        mode
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
        max_sparsity_per_layer
            The max sparsity per layer constraint, to prevent all weight in a layer be pruned in global mode.
        dummy_input
            The dummy input to analyze the topology constraints. Note that, the dummy_input
            should on the same device with the model.
        """
        self.mode = mode
        self.max_sparsity_per_layer = max_sparsity_per_layer
        self.dummy_input = dummy_input
        self.trainer = trainer
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_batches = training_batches
        super().__init__(model, config_list)

    def validate_config(self, model: Module, config_list: List[Dict]):
        schema = PrunerSchema([{
            SchemaOptional('sparsity'): And(float, lambda n: 0 < n < 1),
            'op_types': ['Conv2d'],
            SchemaOptional('op_names'): [str],
            SchemaOptional('exclude'): bool
        }], model, _logger)

        schema.validate(config_list)

    def _collector(self, buffer: List, weight_tensor: Tensor) -> Callable[[Module, Tensor, Tensor], None]:
        def collect_taylor(grad: Tensor):
            if len(buffer) < self.training_batches:
                buffer.append(self._calculate_taylor_expansion(weight_tensor, grad))
        return collect_taylor

    def _calculate_taylor_expansion(self, weight_tensor: Tensor, grad: Tensor) -> Tensor:
        return (weight_tensor.detach() * grad.detach()).data.pow(2)

    def _reset_tools(self):
        hook_targets = {layer_info.name: layer_info.module.weight for layer_info, _ in self._detect_modules_to_compress()}
        collector_info = HookCollectorInfo(hook_targets, 'tensor', self._collector)
        if self.data_collector is None:
            self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer, self.criterion,
                                                                      1, collector_infos=[collector_info])
        else:
            self.data_collector.reset()
        if self.metrics_calculator is None:
            self.metrics_calculator = MultiDataNormMetricsCalculator(p=1, dim=0)
        if self.sparsity_allocator is None:
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self, 0, self.max_sparsity_per_layer)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = Conv2dDependencyAwareAllocator(self, 0, self.dummy_input)
            else:
                raise NotImplementedError('Only support mode `normal`, `global` and `dependency_aware`')
