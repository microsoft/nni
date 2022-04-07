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
    BankSparsityAllocator,
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
    r"""
    This is a basic pruner, and in some papers called it magnitude pruning or fine-grained pruning.
    It will mask the smallest magnitude weights in each specified layer by a saprsity ratio configured in the config list.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Operation types to be pruned.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode : str
        'normal' or 'balance'.
        If setting 'normal' mode, target tensor will be pruned in the way of finegrained pruning.
        If setting 'balance' mode, a specal sparse pattern will chosen by pruner. Take linear
        operation an example, weight tensor will be split into sub block whose shape is aligned to
        balance_gran. Then finegrained pruning will be applied internal of sub block. This sparsity
        pattern have more chance to achieve better trade-off between model performance and hardware
        acceleration. Please refer to releated paper for further information `Balanced Sparsity for
        Efficient DNN Inference on GPU <https://arxiv.org/pdf/1811.00206.pdf>`__.
    balance_gran : list
        Balance_gran is for special sparse pattern balanced sparsity, Default value is None which means pruning
        without awaring balance, namely normal finegrained pruning.
        If passing list of int, LevelPruner will prune the model in the granularity of multi-dimension block.
        Attention that the length of balance_gran should be smaller than tensor dimension.
        For instance, in Linear operation, length of balance_gran should be equal or smaller than two since
        dimension of pruning weight is two. If setting balbance_gran = [5, 5], sparsity = 0.6, pruner will
        divide pruning parameters into multiple block with tile size (5,5) and each bank has 5 * 5 values
        and 10 values would be kept after pruning. Finegrained pruning is applied in the granularity of block
        so that each block will kept same number of non-zero values after pruning. Such pruning method "balance"
        the non-zero value in tensor which create chance for better hardware acceleration.

        Note: If length of given balance_gran smaller than length of pruning tensor shape, it will be made up
              in right align(such as example 1).

            example 1:
                operation: Linear
                pruning tensor: weight
                pruning tensor shape: [32, 32]
                sparsity: 50%
                balance_gran: [4]

                pruning result: Weight tensor whose shape is [32, 32] will be split into 256 [1, 4] sub blocks.
                                Each sub block will be pruned 2 values.

            example 2:
                operation: Linear
                pruning tensor: weight
                pruning tensor shape: [64, 64]
                sparsity: 25%
                balance_gran: [32, 32]

                pruning result: Weight tensor whose shape is [64, 64] will be split into 4 [32, 32] sub blocks.
                                Each sub block will be pruned 256 values.

    Examples
    --------
        >>> model = ...
        >>> from nni.compression.pytorch.pruning import LevelPruner
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
        >>> pruner = LevelPruner(model, config_list)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/level_pruning_torch.py <examples/model_compress/pruning/level_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict], mode: str = "normal", balance_gran: Optional[List] = None):
        self.mode = mode
        self.balance_gran = balance_gran
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
            if self.mode == "normal":
                self.sparsity_allocator = NormalSparsityAllocator(self)
            elif self.mode == "balance":
                assert self.balance_gran is not None, 'balance_gran should be passed as param in balance mode'
                self.sparsity_allocator = BankSparsityAllocator(self, self.balance_gran)
            else:
                raise NotImplementedError('Only support mode `normal` and `balance`')

class NormPruner(BasicPruner):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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
    r"""
    L1 norm pruner computes the l1 norm of the layer weight on the first dimension,
    then prune the weight blocks on this dimension with smaller l1 norm values.
    i.e., compute the l1 norm of the filters in convolution layer as metric values,
    compute the l1 norm of the weight by rows in linear layer as metric values.

    For more details, please refer to `PRUNING FILTERS FOR EFFICIENT CONVNETS <https://arxiv.org/abs/1608.08710>`__.

    In addition, L1 norm pruner also supports dependency-aware mode.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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
    r"""
    L2 norm pruner is a variant of L1 norm pruner.
    The only different between L2 norm pruner and L1 norm pruner is L2 norm pruner prunes the weight with the smallest L2 norm of the weights.

    L2 norm pruner also supports dependency-aware mode.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in L2NormPruner.
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

    Examples
    --------
        >>> model = ...
        >>> from nni.compression.pytorch.pruning import L2NormPruner
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = L2NormPruner(model, config_list)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/norm_pruning_torch.py <examples/model_compress/pruning/norm_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        super().__init__(model, config_list, 2, mode, dummy_input)


class FPGMPruner(BasicPruner):
    r"""
    FPGM pruner prunes the blocks of the weight on the first dimension with the smallest geometric median.
    FPGM chooses the weight blocks with the most replaceable contribution.

    For more details, please refer to `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`__.

    FPGM pruner also supports dependency-aware mode.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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

    Examples
    --------
        >>> model = ...
        >>> from nni.compression.pytorch.pruning import FPGMPruner
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = FPGMPruner(model, config_list)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/fpgm_pruning_torch.py <examples/model_compress/pruning/fpgm_pruning_torch.py>`
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
    r"""
    Slim pruner adds sparsity regularization on the scaling factors of batch normalization (BN) layers during training to identify unimportant channels.
    The channels with small scaling factor values will be pruned.

    For more details, please refer to `Learning Efficient Convolutional Networks through Network Slimming <https://arxiv.org/abs/1708.06519>`__\.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
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

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import SlimPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
        >>> pruner = SlimPruner(model, config_list, trainer, traced_optimizer, criterion, training_epochs=1)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/slim_pruning_torch.py <examples/model_compress/pruning/slim_pruning_torch.py>`
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
        Model to be pruned.
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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
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
    r"""
    Activation APoZ rank pruner is a pruner which prunes on the first weight dimension,
    with the smallest importance criterion ``APoZ`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.
    The pruning criterion ``APoZ`` is explained in the paper `Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures <https://arxiv.org/abs/1607.03250>`__.

    The APoZ is defined as:
    :math:`APoZ_{c}^{(i)} = APoZ\left(O_{c}^{(i)}\right)=\frac{\sum_{k}^{N} \sum_{j}^{M} f\left(O_{c, j}^{(i)}(k)=0\right)}{N \times M}`

    Activation APoZ rank pruner also supports dependency-aware mode.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
    config_list : List[Dict]
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in ActivationAPoZRankPruner.
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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``..
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

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import ActivationAPoZRankPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = ActivationAPoZRankPruner(model, config_list, trainer, traced_optimizer, criterion, training_batches=20)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/activation_pruning_torch.py <examples/model_compress/pruning/activation_pruning_torch.py>`
    """
    def _activation_trans(self, output: Tensor) -> Tensor:
        # return a matrix that the position of zero in `output` is one, others is zero.
        return torch.eq(self._activation(output.detach()), torch.zeros_like(output)).type_as(output)

    def _get_metrics_calculator(self) -> MetricsCalculator:
        return APoZRankMetricsCalculator(dim=1)


class ActivationMeanRankPruner(ActivationPruner):
    r"""
    Activation mean rank pruner is a pruner which prunes on the first weight dimension,
    with the smallest importance criterion ``mean activation`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.

    The pruning criterion ``mean activation`` is explained in section 2.2 of the paper `Pruning Convolutional Neural Networks for Resource Efficient Inference <https://arxiv.org/abs/1611.06440>`__.

    Activation mean rank pruner also supports dependency-aware mode.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``..
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

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import ActivationMeanRankPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = ActivationMeanRankPruner(model, config_list, trainer, traced_optimizer, criterion, training_batches=20)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/activation_pruning_torch.py <examples/model_compress/pruning/activation_pruning_torch.py>`
    """
    def _activation_trans(self, output: Tensor) -> Tensor:
        # return the activation of `output` directly.
        return self._activation(output.detach())

    def _get_metrics_calculator(self) -> MetricsCalculator:
        return MeanRankMetricsCalculator(dim=1)


class TaylorFOWeightPruner(BasicPruner):
    r"""
    Taylor FO weight pruner is a pruner which prunes on the first weight dimension,
    based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity.
    The estimated importance is defined as the paper `Importance Estimation for Neural Network Pruning <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__.

    :math:`\widehat{\mathcal{I}}_{\mathcal{S}}^{(1)}(\mathbf{W}) \triangleq \sum_{s \in \mathcal{S}} \mathcal{I}_{s}^{(1)}(\mathbf{W})=\sum_{s \in \mathcal{S}}\left(g_{s} w_{s}\right)^{2}`

    Taylor FO weight pruner also supports dependency-aware mode.

    What's more, we provide a global-sort mode for this pruner which is aligned with paper implementation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be pruned.
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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
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

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import TaylorFOWeightPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = TaylorFOWeightPruner(model, config_list, trainer, traced_optimizer, criterion, training_batches=20)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/taylorfo_pruning_torch.py <examples/model_compress/pruning/taylorfo_pruning_torch.py>`
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
        hook_targets = {name: wrapper.weight for name, wrapper in self.get_modules_wrapper().items()}
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
    r"""
    Alternating Direction Method of Multipliers (ADMM) is a mathematical optimization technique,
    by decomposing the original nonconvex problem into two subproblems that can be solved iteratively.
    In weight pruning problem, these two subproblems are solved via 1) gradient descent algorithm and 2) Euclidean projection respectively. 

    During the process of solving these two subproblems, the weights of the original model will be changed.
    Then a fine-grained pruning will be applied to prune the model according to the config list given.

    This solution framework applies both to non-structured and different variations of structured pruning schemes.

    For more details, please refer to `A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers <https://arxiv.org/abs/1804.03294>`__.

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
        E.g. ``traced_optimizer = nni.trace(torch.nn.Adam)(model.parameters())``.
    criterion : Callable[[Tensor, Tensor], Tensor]
        The criterion function used in trainer. Take model output and target value as input, and return the loss.
    iterations : int
        The total iteration number in admm pruning algorithm.
    training_epochs : int
        The epoch number for training model in each iteration.
    granularity : str
        'fine-grained' or 'coarse-grained'.
        If 'coarse-grained' is set, ADMM pruner will generate masks on output channels wise.
        In original admm pruning paper, author implemented a fine-grained admm pruning.
        In auto-compress paper, author used coarse-grained admm pruning.

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import ADMMPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = ADMMPruner(model, config_list, trainer, traced_optimizer, criterion, iterations=10, training_epochs=1)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/admm_pruning_torch.py <examples/model_compress/pruning/admm_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor], iterations: int,
                 training_epochs: int, granularity: str = 'fine-grained'):
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.iterations = iterations
        self.training_epochs = training_epochs
        assert granularity in ['fine-grained', 'coarse-grained']
        self.granularity = granularity
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
            if self.granularity == 'fine-grained':
                self.metrics_calculator = NormMetricsCalculator(p=1)
            elif self.granularity == 'coarse-grained':
                self.metrics_calculator = NormMetricsCalculator(dim=0, p=1)
        if self.sparsity_allocator is None:
            if self.granularity == 'fine-grained':
                self.sparsity_allocator = NormalSparsityAllocator(self)
            elif self.granularity == 'coarse-grained':
                self.sparsity_allocator = NormalSparsityAllocator(self, dim=0)

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
