# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
import functools
import logging
from typing import List, Dict, Tuple, Callable, Optional, overload

from schema import And, Or, Optional as SchemaOptional, SchemaError
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer

from ..base import Pruner

from .tools import (
    DataCollector,
    HookCollectorInfo,
    TargetDataCollector,
    EvaluatorBasedTargetDataCollector,
    EvaluatorBasedHookDataCollector
)

# TODO: remove in nni v3.0.
from .tools import (
    WeightTrainerBasedDataCollector,
    SingleHookTrainerBasedDataCollector
)

from .tools import (
    MetricsCalculator,
    NormMetricsCalculator,
    HookDataNormMetricsCalculator,
    DistMetricsCalculator,
    APoZRankMetricsCalculator,
    MeanRankMetricsCalculator
)

from .tools import (
    SparsityAllocator,
    NormalSparsityAllocator,
    BankSparsityAllocator,
    GlobalSparsityAllocator,
    DependencyAwareAllocator
)

from ..utils import (
    CompressorSchema,
    OptimizerConstructHelper,
    Scaling,
    Evaluator,
    LightningEvaluator,
    TorchEvaluator,
    ForwardHook,
    TensorHook,
    config_list_canonical
)

from ..utils.docstring import _EVALUATOR_DOCSTRING

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
    data_collector: DataCollector
    metrics_calculator: MetricsCalculator
    sparsity_allocator: SparsityAllocator

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
        err_msg = 'Model and/or config_list are not set in this pruner, please set them by reset() before compress().'
        assert self.bound_model is not None and self.config_list is not None, err_msg
        assert self.data_collector is not None and self.metrics_calculator is not None and self.sparsity_allocator is not None
        data = self.data_collector.collect()
        _logger.debug('Collected Data:\n%s', data)
        metrics = self.metrics_calculator.calculate_metrics(data)
        _logger.debug('Metrics Calculate:\n%s', metrics)
        masks = self.sparsity_allocator.generate_sparsity(metrics)
        _logger.debug('Masks:\n%s', masks)
        self.load_masks(masks)
        return self.bound_model, masks


_LEGACY_TRAINER = Callable[[Module, Optimizer, Callable], None]
_LEGACY_CRITERION = Callable[[Tensor, Tensor], Tensor]


# TODO: remove in nni v3.0.
class EvaluatorBasedPruner(BasicPruner):
    evaluator: LightningEvaluator | TorchEvaluator
    using_evaluator: bool
    trainer: _LEGACY_TRAINER
    traced_optimizer: Optimizer
    criterion: _LEGACY_CRITERION

    def _init_evaluator(self, model: Module, new_api: List[str], old_api: List[str], init_kwargs: Dict, args: Tuple,
                        kwargs: Dict) -> Dict:
        # for fake __init__ overload, parsing args and kwargs, initializing evaluator or [trainer, traced_optimizer, criterion],
        # return the remaining arguments.
        if (len(args) > 0 and isinstance(args[0], Evaluator)) or (len(args) == 0 and isinstance(kwargs.get('evaluator', None), Evaluator)):
            init_kwargs = self._parse_args(new_api, args, kwargs, init_kwargs)
            self.evaluator: LightningEvaluator | TorchEvaluator = init_kwargs.pop('evaluator')
            if not self.evaluator._initialization_complete:
                self.evaluator._init_optimizer_helpers(model)  # type: ignore
            self.using_evaluator = True
        else:
            init_kwargs = self._parse_args(old_api, args, kwargs, init_kwargs)
            self.trainer: _LEGACY_TRAINER = init_kwargs.pop('trainer')
            traced_optimizer: Optimizer | OptimizerConstructHelper = init_kwargs.pop('traced_optimizer')
            self.criterion: _LEGACY_CRITERION = init_kwargs.pop('criterion')
            if isinstance(traced_optimizer, OptimizerConstructHelper):
                self.optimizer_helper = traced_optimizer
            else:
                self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
            self.using_evaluator = False
            warn_msg = f"The old API ...{','.join(old_api)} will be deprecated after NNI v3.0, " + \
                       "please using the new one ...{','.join(new_api)}"
            _logger.warning(warn_msg)
        return init_kwargs

    def _parse_args(self, arg_names: List, args: Tuple, kwargs: Dict, def_kwargs: Dict) -> Dict:
        merged_kwargs = {arg_names[idx]: arg for idx, arg in enumerate(args)}
        for key, value in kwargs.items():
            if key in merged_kwargs:
                raise TypeError(f"{self.__class__.__name__}.__init__() got multiple values for argument '{key}'")
            merged_kwargs[key] = value
        for key, value in def_kwargs.items():
            if key not in merged_kwargs and key in arg_names:
                merged_kwargs[key] = value
        diff = set(arg_names).difference(merged_kwargs.keys())
        if diff:
            raise TypeError(f"{self.__class__.__name__}.__init__() missing {len(diff)} required positional argument: {diff}")
        diff = set(merged_kwargs.keys()).difference(arg_names)
        if diff:
            raise TypeError(f"{self.__class__.__name__}.__init__() got {len(diff)} unexpected keyword argument: {diff}")
        return merged_kwargs

    def compress(self) -> Tuple[Module, Dict]:
        result = super().compress()
        if self.using_evaluator:
            self.evaluator.unbind_model()
        return result


class LevelPruner(BasicPruner):
    r"""
    This is a basic pruner, and in some papers called it magnitude pruning or fine-grained pruning.
    It will mask the smallest magnitude weights in each specified layer by a saprsity ratio configured in the config list.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Operation types to be pruned.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode
        'normal' or 'balance'.
        If setting 'normal' mode, target tensor will be pruned in the way of finegrained pruning.
        If setting 'balance' mode, a specal sparse pattern will chosen by pruner. Take linear
        operation an example, weight tensor will be split into sub block whose shape is aligned to
        balance_gran. Then finegrained pruning will be applied internal of sub block. This sparsity
        pattern have more chance to achieve better trade-off between model performance and hardware
        acceleration. Please refer to releated paper for further information `Balanced Sparsity for
        Efficient DNN Inference on GPU <https://arxiv.org/pdf/1811.00206.pdf>`__.
    balance_gran
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

    For detailed example please refer to
    :githublink:`examples/model_compress/pruning/level_pruning_torch.py <examples/model_compress/pruning/level_pruning_torch.py>`
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
        if not hasattr(self, 'data_collector'):
            self.data_collector = TargetDataCollector(self)
        else:
            self.data_collector.reset()
        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = NormMetricsCalculator()
        if not hasattr(self, 'sparsity_allocator'):
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
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
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
        scalers = Scaling(kernel_size=[1], kernel_padding_mode='back')
        if not hasattr(self, 'sparsity_allocator'):
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, scalers)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = DependencyAwareAllocator(self, self.dummy_input, scalers)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')
        if not hasattr(self, 'data_collector'):
            self.data_collector = TargetDataCollector(self)
        else:
            self.data_collector.reset()
        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = NormMetricsCalculator(p=self.p, scalers=scalers)


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
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in L1NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
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

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        super().__init__(model, config_list, 1, mode, dummy_input)


class L2NormPruner(NormPruner):
    r"""
    L2 norm pruner is a variant of L1 norm pruner.
    The only different between L2 norm pruner and L1 norm pruner is
    L2 norm pruner prunes the weight with the smallest L2 norm of the weights.

    L2 norm pruner also supports dependency-aware mode.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in L2NormPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
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

    Examples
    --------
        >>> model = ...
        >>> from nni.compression.pytorch.pruning import L2NormPruner
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = L2NormPruner(model, config_list)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to
    :githublink:`examples/model_compress/pruning/norm_pruning_torch.py <examples/model_compress/pruning/norm_pruning_torch.py>`
    """

    def __init__(self, model: Module, config_list: List[Dict],
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        super().__init__(model, config_list, 2, mode, dummy_input)


class FPGMPruner(BasicPruner):
    r"""
    FPGM pruner prunes the blocks of the weight on the first dimension with the smallest geometric median.
    FPGM chooses the weight blocks with the most replaceable contribution.

    For more details, please refer to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`__.

    FPGM pruner also supports dependency-aware mode.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in FPGMPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    mode
        'normal' or 'dependency_aware'.
        If prune the model in a dependency-aware way, this pruner will
        prune the model according to the FPGM of weights and the channel-dependency or
        group-dependency of the model. In this way, the pruner will force the conv layers
        that have dependencies to prune the same channels, so the speedup module can better
        harvest the speed benefit from the pruned model. Note that, if set 'dependency_aware'
        , the dummy_input cannot be None, because the pruner needs a dummy input to trace the
        dependency between the conv layers.
    dummy_input
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.

    Examples
    --------
        >>> model = ...
        >>> from nni.compression.pytorch.pruning import FPGMPruner
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = FPGMPruner(model, config_list)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to
    :githublink:`examples/model_compress/pruning/fpgm_pruning_torch.py <examples/model_compress/pruning/fpgm_pruning_torch.py>`
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
        scalers = Scaling(kernel_size=[1], kernel_padding_mode='back')
        if not hasattr(self, 'sparsity_allocator'):
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, scalers)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = DependencyAwareAllocator(self, self.dummy_input, scalers)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')
        if not hasattr(self, 'data_collector'):
            self.data_collector = TargetDataCollector(self)
        else:
            self.data_collector.reset()
        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = DistMetricsCalculator(p=2, scalers=scalers)


class SlimPruner(EvaluatorBasedPruner):
    __doc__ = r"""Slim pruner adds sparsity regularization on the scaling factors of batch normalization (BN) layers during training
    to identify unimportant channels. The channels with small scaling factor values will be pruned.

    For more details, please refer to `Learning Efficient Convolutional Networks through Network Slimming <https://arxiv.org/abs/1708.06519>`__\.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - total_sparsity : This is to specify the total sparsity for all layers in this config, each layer may have different sparsity.
            - max_sparsity_per_layer : Always used with total_sparsity. Limit the max sparsity of each layer.
            - op_types : Only BatchNorm2d is supported in SlimPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    training_epochs
        The epoch number for training model to sparsify the BN weight.
    scale
        Penalty parameter for sparsification, which could reduce overfitting.
    mode
        'normal' or 'global'.
        If prune the model in a global way, all layer weights with same config will be considered uniformly.
        That means a single layer may not reach or exceed the sparsity setting in config,
        but the total pruned weights meet the sparsity setting.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/slim_pruning_torch.py <examples/model_compress/pruning/slim_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: LightningEvaluator | TorchEvaluator,
                 training_epochs: int, scale: float = 0.0001, mode='global'):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], trainer: _LEGACY_TRAINER, traced_optimizer: Optimizer,
                 criterion: _LEGACY_CRITERION, training_epochs: int, scale: float = 0.0001, mode='global'):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'training_epochs', 'scale', 'mode']
        old_api = ['trainer', 'traced_optimizer', 'criterion', 'training_epochs', 'scale', 'mode']
        init_kwargs = {'scale': 0.0001, 'mode': 'global'}
        init_kwargs = self._init_evaluator(model, new_api, old_api, init_kwargs, args, kwargs)

        self.training_epochs, self._scale, self.mode = init_kwargs['training_epochs'], init_kwargs['scale'], init_kwargs['mode']

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
                err_msg = '`config_list` validation failed. If global mode is set in this pruner, ' + \
                          '`sparsity_per_layer` and `sparsity` are not supported, make sure `total_sparsity` is set in config_list.'
                _logger.error(err_msg)
            raise e

    # TODO: remove in nni v3.0.
    def criterion_patch(self, criterion: Callable[[Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor], Tensor]:
        def patched_criterion(input_tensor: Tensor, target: Tensor):
            sum_l1 = 0
            for wrapper in self.get_modules_wrapper().values():
                sum_l1 += torch.norm(wrapper.weight, p=1)  # type: ignore
            return criterion(input_tensor, target) + self._scale * sum_l1
        return patched_criterion

    def loss_patch(self, origin_loss: Tensor):
        # additional weight norm loss in Slim, used to sparse the weight value.
        sum_l1 = 0
        for wrapper in self.get_modules_wrapper().values():
            target_name = 'weight'
            sum_l1 += torch.norm(getattr(wrapper, target_name), p=1)  # type: ignore
        return self._scale * sum_l1 + origin_loss

    def reset_tools(self):
        if self.using_evaluator:
            # TODO: move to other place in nni v3.0
            self.evaluator.unbind_model()
            self.evaluator.bind_model(self.bound_model, self.get_origin2wrapped_parameter_name_map())  # type: ignore
            if not hasattr(self, 'data_collector'):
                self.data_collector = EvaluatorBasedTargetDataCollector(self, self.evaluator, loss_patch=self.loss_patch,
                                                                        max_epochs=self.training_epochs)
            else:
                self.data_collector.reset(loss_patch=self.loss_patch)
        else:
            if not hasattr(self, 'data_collector'):
                self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                      self.training_epochs, criterion_patch=self.criterion_patch)
            else:
                self.data_collector.reset()

        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = NormMetricsCalculator()
        if not hasattr(self, 'sparsity_allocator'):
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self)
            else:
                raise NotImplementedError('Only support mode `normal` and `global`')


class ActivationPruner(EvaluatorBasedPruner):
    __doc__ = r"""Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in ActivationPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    training_steps
        The step number used to collect activations.
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
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: LightningEvaluator | TorchEvaluator, training_steps: int,
                 activation: str = 'relu', mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], trainer: _LEGACY_TRAINER, traced_optimizer: Optimizer,
                 criterion: _LEGACY_CRITERION, training_batches: int, activation: str = 'relu',
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'training_steps', 'activation', 'mode', 'dummy_input']
        old_api = ['trainer', 'traced_optimizer', 'criterion', 'training_batches', 'activation', 'mode', 'dummy_input']
        init_kwargs = {'activation': 'relu', 'mode': 'normal', 'dummy_input': None}
        init_kwargs = self._init_evaluator(model, new_api, old_api, init_kwargs, args, kwargs)

        self.training_steps: int = init_kwargs.get('training_steps', init_kwargs.get('training_batches'))
        self._activation: Callable[[Tensor], Tensor] = self._choose_activation(init_kwargs['activation'])
        self.mode: str = init_kwargs['mode']
        self.dummy_input = init_kwargs['dummy_input']

        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for sub_shcema in schema_list:
            sub_shcema[SchemaOptional('op_types')] = ['Conv2d', 'Linear']
        schema = CompressorSchema(schema_list, model, _logger)

        schema.validate(config_list)

    def _choose_activation(self, activation: str = 'relu') -> Callable:
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'relu6':
            return F.relu6
        else:
            raise Exception('Unsupported activatoin {}'.format(activation))

    def _collector(self, buffer: List) -> Callable[[Module, Tensor, Tensor], None]:
        assert len(buffer) == 0, 'Buffer pass to activation pruner collector is not empty.'
        # The length of the buffer used in this pruner will always be 2.
        # buffer[0] is the number of how many batches are counted in buffer[1].
        # buffer[1] is a tensor and the size of buffer[1] is same as the activation.
        buffer.append(0)

        def collect_activation(_module: Module, _input: Tensor, output: Tensor):
            activation = self._activation_trans(output)
            if len(buffer) == 1:
                buffer.append(torch.zeros_like(activation))
            if buffer[0] < self.training_steps:
                buffer[1] += activation
                buffer[0] += 1
        return collect_activation

    def _activation_trans(self, output: Tensor) -> Tensor:
        raise NotImplementedError()

    def reset_tools(self):
        scalers = Scaling(kernel_size=[1], kernel_padding_mode='back')
        if not hasattr(self, 'sparsity_allocator'):
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, scalers)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = DependencyAwareAllocator(self, self.dummy_input, scalers)
            else:
                raise NotImplementedError('Only support mode `normal` and `dependency_aware`')

        if self.using_evaluator:
            # TODO: move to other place in nni v3.0
            self.evaluator.unbind_model()
            self.evaluator.bind_model(self.bound_model, self.get_origin2wrapped_parameter_name_map())  # type: ignore
            forward_hooks = {}
            for module_name, wrapper in self.get_modules_wrapper().items():
                target_name = 'weight'
                forward_hooks[module_name] = {target_name: ForwardHook(wrapper, module_name, self._collector)}
            if not hasattr(self, 'data_collector'):
                self.data_collector = EvaluatorBasedHookDataCollector(self, self.evaluator, hooks=forward_hooks,
                                                                      max_steps=self.training_steps)
            else:
                self.data_collector.reset(hooks=forward_hooks)
        else:
            collector_info = HookCollectorInfo([layer_info for layer_info, _ in self._detect_modules_to_compress()],
                                               'forward', self._collector)
            if not hasattr(self, 'data_collector'):
                self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                          1, collector_infos=[collector_info])
            else:
                self.data_collector.reset([collector_info])  # type: ignore

        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = self._create_metrics_calculator()

    def _create_metrics_calculator(self) -> MetricsCalculator:
        raise NotImplementedError()


class ActivationAPoZRankPruner(ActivationPruner):
    __doc__ = r"""Activation APoZ rank pruner is a pruner which prunes on the first weight dimension,
    with the smallest importance criterion ``APoZ`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.
    The pruning criterion ``APoZ`` is explained in the paper `Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures <https://arxiv.org/abs/1607.03250>`__.

    The APoZ is defined as:
    :math:`APoZ_{c}^{(i)} = APoZ\left(O_{c}^{(i)}\right)=\frac{\sum_{k}^{N} \sum_{j}^{M} f\left(O_{c, j}^{(i)}(k)=0\right)}{N \times M}`
    """ + r"""

    Activation APoZ rank pruner also supports dependency-aware mode.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in ActivationAPoZRankPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    training_steps
        The step number used to collect activations.
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

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/activation_pruning_torch.py <examples/model_compress/pruning/activation_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def _activation_trans(self, output: Tensor) -> Tensor:
        # return a matrix that the position of zero in `output` is one, others is zero.
        return torch.eq(self._activation(output.detach()), torch.zeros_like(output)).type_as(output).mean(0)

    def _create_metrics_calculator(self) -> MetricsCalculator:
        return APoZRankMetricsCalculator(Scaling(kernel_size=[1], kernel_padding_mode='back'))


class ActivationMeanRankPruner(ActivationPruner):
    __doc__ = r"""
    Activation mean rank pruner is a pruner which prunes on the first weight dimension,
    with the smallest importance criterion ``mean activation`` calculated from the output activations of convolution layers to achieve a preset level of network sparsity.

    The pruning criterion ``mean activation`` is explained in section 2.2 of the paper `Pruning Convolutional Neural Networks for Resource Efficient Inference <https://arxiv.org/abs/1611.06440>`__.

    Activation mean rank pruner also supports dependency-aware mode.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - op_types : Conv2d and Linear are supported in ActivationPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    training_steps
        The step number used to collect activations.
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

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/activation_pruning_torch.py <examples/model_compress/pruning/activation_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def _activation_trans(self, output: Tensor) -> Tensor:
        # return the activation of `output` directly.
        return self._activation(output.detach()).mean(0)

    def _create_metrics_calculator(self) -> MetricsCalculator:
        return MeanRankMetricsCalculator(Scaling(kernel_size=[1], kernel_padding_mode='back'))


class TaylorFOWeightPruner(EvaluatorBasedPruner):
    __doc__ = r"""
    Taylor FO weight pruner is a pruner which prunes on the first weight dimension,
    based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity.
    The estimated importance is defined as the paper `Importance Estimation for Neural Network Pruning <http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf>`__.

    :math:`\widehat{\mathcal{I}}_{\mathcal{S}}^{(1)}(\mathbf{W}) \triangleq \sum_{s \in \mathcal{S}} \mathcal{I}_{s}^{(1)}(\mathbf{W})=\sum_{s \in \mathcal{S}}\left(g_{s} w_{s}\right)^{2}`
    """ + r"""

    Taylor FO weight pruner also supports dependency-aware mode.

    What's more, we provide a global-sort mode for this pruner which is aligned with paper implementation.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - total_sparsity : This is to specify the total sparsity for all layers in this config, each layer may have different sparsity.
            - max_sparsity_per_layer : Always used with total_sparsity. Limit the max sparsity of each layer.
            - op_types : Conv2d and Linear are supported in TaylorFOWeightPruner.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    training_steps
        The step number used to collect activations.
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
    dummy_input
        The dummy input to analyze the topology constraints. Note that, the dummy_input
        should on the same device with the model.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/taylorfo_pruning_torch.py <examples/model_compress/pruning/taylorfo_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: LightningEvaluator | TorchEvaluator, training_steps: int,
                 mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], trainer: _LEGACY_TRAINER, traced_optimizer: Optimizer,
                 criterion: _LEGACY_CRITERION, training_batches: int, mode: str = 'normal', dummy_input: Optional[Tensor] = None):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'training_steps', 'mode', 'dummy_input']
        old_api = ['trainer', 'traced_optimizer', 'criterion', 'training_batches', 'mode', 'dummy_input']
        init_kwargs = {'mode': 'normal', 'dummy_input': None}
        init_kwargs = self._init_evaluator(model, new_api, old_api, init_kwargs, args, kwargs)

        self.training_steps: int = init_kwargs.get('training_steps', init_kwargs.get('training_batches'))
        self.mode: str = init_kwargs['mode']
        self.dummy_input = init_kwargs['dummy_input']

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
                err_msg = '`config_list` validation failed. If global mode is set in this pruner, ' + \
                          '`sparsity_per_layer` and `sparsity` are not supported, make sure `total_sparsity` is set in config_list.'
                _logger.error(err_msg)
            raise e

    def _collector(self, buffer: List, weight_tensor: Tensor) -> Callable[[Tensor], None]:
        assert len(buffer) == 0, 'Buffer pass to taylor pruner collector is not empty.'
        buffer.append(0)

        def collect_taylor(grad: Tensor):
            if len(buffer) == 1:
                buffer.append(torch.zeros_like(grad))
            if buffer[0] < self.training_steps:
                buffer[1] += self._calculate_taylor_expansion(weight_tensor, grad)
                buffer[0] += 1
        return collect_taylor

    def _calculate_taylor_expansion(self, weight_tensor: Tensor, grad: Tensor) -> Tensor:
        return (weight_tensor.detach() * grad.detach()).data.pow(2)

    def reset_tools(self):
        scalers = Scaling(kernel_size=[1], kernel_padding_mode='back')
        if not hasattr(self, 'sparsity_allocator'):
            if self.mode == 'normal':
                self.sparsity_allocator = NormalSparsityAllocator(self, scalers)
            elif self.mode == 'global':
                self.sparsity_allocator = GlobalSparsityAllocator(self, scalers)
            elif self.mode == 'dependency_aware':
                self.sparsity_allocator = DependencyAwareAllocator(self, self.dummy_input, scalers)
            else:
                raise NotImplementedError('Only support mode `normal`, `global` and `dependency_aware`')

        if self.using_evaluator:
            # TODO: move to other place in nni v3.0
            self.evaluator.unbind_model()
            self.evaluator.bind_model(self.bound_model, self.get_origin2wrapped_parameter_name_map())  # type: ignore
            tensor_hooks = {}
            for module_name, wrapper in self.get_modules_wrapper().items():
                target_name = 'weight'
                target = getattr(wrapper, target_name)
                tensor_hooks[module_name] = {target_name: TensorHook(target, module_name,
                                                                     functools.partial(self._collector, weight_tensor=target))}
            if not hasattr(self, 'data_collector'):
                self.data_collector = EvaluatorBasedHookDataCollector(self, self.evaluator, hooks=tensor_hooks,
                                                                      max_steps=self.training_steps)
            else:
                self.data_collector.reset(hooks=tensor_hooks)
        else:
            hook_targets = {name: wrapper.weight for name, wrapper in self.get_modules_wrapper().items()}  # type: ignore
            collector_info = HookCollectorInfo(hook_targets, 'tensor', self._collector)  # type: ignore
            if not hasattr(self, 'data_collector'):
                self.data_collector = SingleHookTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                          1, collector_infos=[collector_info])
            else:
                self.data_collector.reset([collector_info])  # type: ignore

        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = HookDataNormMetricsCalculator(p=1, scalers=scalers)


class ADMMPruner(EvaluatorBasedPruner):
    __doc__ = r"""
    Alternating Direction Method of Multipliers (ADMM) is a mathematical optimization technique,
    by decomposing the original nonconvex problem into two subproblems that can be solved iteratively.
    In weight pruning problem, these two subproblems are solved via 1) gradient descent algorithm and 2) Euclidean projection respectively. 

    During the process of solving these two subproblems, the weights of the original model will be changed.
    Then a fine-grained pruning will be applied to prune the model according to the config list given.

    This solution framework applies both to non-structured and different variations of structured pruning schemes.

    For more details, please refer to `A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers <https://arxiv.org/abs/1804.03294>`__.

    Parameters
    ----------
    model
        Model to be pruned.
    config_list
        Supported keys:
            - sparsity : This is to specify the sparsity for each layer in this config to be compressed.
            - sparsity_per_layer : Equals to sparsity.
            - rho : Penalty parameters in ADMM algorithm.
            - op_types : Operation types to be pruned.
            - op_names : Operation names to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    iterations
        The total iteration number in admm pruning algorithm.
    training_epochs
        The epoch number for training model in each iteration.
    granularity
        'fine-grained' or 'coarse-grained'.
        If 'coarse-grained' is set, ADMM pruner will generate masks on output channels wise.
        In original admm pruning paper, author implemented a fine-grained admm pruning.
        In auto-compress paper, author used coarse-grained admm pruning.

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/admm_pruning_torch.py <examples/model_compress/pruning/admm_pruning_torch.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: LightningEvaluator | TorchEvaluator, iterations: int,
                 training_epochs: int, granularity: str = 'fine-grained'):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], trainer: _LEGACY_TRAINER,
                 traced_optimizer: Optimizer, criterion: _LEGACY_CRITERION, iterations: int,
                 training_epochs: int, granularity: str = 'fine-grained'):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'iterations', 'training_epochs', 'granularity']
        old_api = ['trainer', 'traced_optimizer', 'criterion', 'iterations', 'training_epochs', 'granularity']
        init_kwargs = {'granularity': 'fine-grained'}
        init_kwargs = self._init_evaluator(model, new_api, old_api, init_kwargs, args, kwargs)

        self.iterations: int = init_kwargs['iterations']
        self.training_epochs: int = init_kwargs['training_epochs']
        assert init_kwargs['granularity'] in ['fine-grained', 'coarse-grained']
        self.granularity: str = init_kwargs['granularity']

        self.Z, self.U = {}, {}
        super().__init__(model, config_list)

    def reset(self, model: Module, config_list: List[Dict]):
        super().reset(model, config_list)
        # FIXME: Only support pruning 'weight' right now.
        target_name = 'weight'
        for module_name, wrapper in self.get_modules_wrapper().items():
            self.Z[module_name] = {target_name: wrapper.weight.data.clone()}  # type: ignore
        self.U = {module_name: {target_name: torch.zeros_like(z[target_name])} for module_name, z in self.Z.items()}

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        for schema in schema_list:
            schema.update({SchemaOptional('rho'): And(float, lambda n: n > 0)})
        schema_list.append(deepcopy(EXCLUDE_SCHEMA))
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    # TODO: remove in nni v3.0.
    def criterion_patch(self, origin_criterion: Callable[[Tensor, Tensor], Tensor]):
        def patched_criterion(output: Tensor, target: Tensor):
            penalty = torch.tensor(0.0).to(output.device)
            for name, wrapper in self.get_modules_wrapper().items():
                rho = wrapper.config.get('rho', 1e-4)
                self.Z[name]['weight'] = self.Z[name]['weight'].to(wrapper.weight.device)  # type: ignore
                self.U[name]['weight'] = self.U[name]['weight'].to(wrapper.weight.device)  # type: ignore
                penalty += (rho / 2) * torch.sqrt(torch.norm(wrapper.weight - self.Z[name]['weight'] + self.U[name]['weight']))
            return origin_criterion(output, target) + penalty
        return patched_criterion

    def loss_patch(self, origin_loss: Tensor):
        penalty = 0
        for name, wrapper in self.get_modules_wrapper().items():
            rho = wrapper.config.get('rho', 1e-4)
            self.Z[name]['weight'] = self.Z[name]['weight'].to(wrapper.weight.device)  # type: ignore
            self.U[name]['weight'] = self.U[name]['weight'].to(wrapper.weight.device)  # type: ignore
            penalty += (rho / 2) * torch.sqrt(torch.norm(wrapper.weight - self.Z[name]['weight'] + self.U[name]['weight']))
        return origin_loss + penalty

    def reset_tools(self):
        if self.using_evaluator:
            # TODO: move to other place in nni v3.0
            self.evaluator.unbind_model()
            self.evaluator.bind_model(self.bound_model, self.get_origin2wrapped_parameter_name_map())  # type: ignore
            if not hasattr(self, 'data_collector'):
                self.data_collector = EvaluatorBasedTargetDataCollector(self, self.evaluator, loss_patch=self.loss_patch,
                                                                        max_epochs=self.training_epochs)
            else:
                self.data_collector.reset(loss_patch=self.loss_patch)
        else:
            if not hasattr(self, 'data_collector'):
                self.data_collector = WeightTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion,
                                                                      self.training_epochs, criterion_patch=self.criterion_patch)
            else:
                self.data_collector.reset()
        if not hasattr(self, 'metrics_calculator'):
            if self.granularity == 'fine-grained':
                self.metrics_calculator = NormMetricsCalculator(p=1)
            elif self.granularity == 'coarse-grained':
                self.metrics_calculator = NormMetricsCalculator(p=1, scalers=Scaling(kernel_size=[1], kernel_padding_mode='back'))
        if not hasattr(self, 'sparsity_allocator'):
            if self.granularity == 'fine-grained':
                self.sparsity_allocator = NormalSparsityAllocator(self)
            elif self.granularity == 'coarse-grained':
                self.sparsity_allocator = NormalSparsityAllocator(self, Scaling(kernel_size=[1], kernel_padding_mode='back'))

    def compress(self) -> Tuple[Module, Dict]:
        assert self.bound_model is not None
        for i in range(self.iterations):
            _logger.info('======= ADMM Iteration %d Start =======', i)
            data = self.data_collector.collect()

            for module_name, targets_data in data.items():
                for target_name, target_data in targets_data.items():
                    self.U[module_name][target_name] = self.U[module_name][target_name].to(target_data.device)
                    self.Z[module_name][target_name] = target_data + self.U[module_name][target_name]
            metrics = self.metrics_calculator.calculate_metrics(self.Z)
            masks = self.sparsity_allocator.generate_sparsity(metrics)

            for module_name, targets_mask in masks.items():
                target_name = 'weight'
                self.Z[module_name][target_name] = self.Z[module_name][target_name].mul(targets_mask[target_name])
                self.U[module_name][target_name] = self.U[module_name][target_name] + data[module_name][target_name] - \
                                                   self.Z[module_name][target_name]

        self.Z, self.U = {}, {}
        torch.cuda.empty_cache()

        metrics = self.metrics_calculator.calculate_metrics(data)  # type: ignore
        masks = self.sparsity_allocator.generate_sparsity(metrics)

        self.load_masks(masks)

        if self.using_evaluator:
            self.evaluator.unbind_model()

        return self.bound_model, masks
