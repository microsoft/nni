# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from typing import Dict, List, Tuple, Callable

import torch
from torch import autograd, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer, Adam

from nni.algorithms.compression.v2.pytorch.base import PrunerModuleWrapper, LayerInfo
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import BasicPruner, NORMAL_SCHEMA, EXCLUDE_SCHEMA, INTERNAL_SCHEMA
from nni.algorithms.compression.v2.pytorch.utils import CompressorSchema, OptimizerConstructHelper
from nni.common.serializer import Traceable

from .tools.base import TrainerBasedDataCollector

from .tools import (
    StraightMetricsCalculator,
    NormalSparsityAllocator
)

_logger = logging.getLogger(__name__)


class PrunerScoredModuleWrapper(PrunerModuleWrapper):
    """
    Wrap a module to enable data parallel, forward method customization and buffer registeration.
    Different from `PrunerModuleWrapper`, `PrunerScoredModuleWrapper` will record the gradient.

    Parameters
    ----------
    module
        The module user wants to compress.
    config
        The configurations that users specify for compression.
    module_name
        The name of the module to compress, wrapper module shares same name.
    """
    def __init__(self, module: Module, module_name: str, config: Dict):
        super().__init__(module, module_name, config)
        self.weight_score = Parameter(torch.empty(self.weight.size()))
        torch.nn.init.constant_(self.weight_score, val=0.0)

    def forward(self, *inputs):
        # apply mask to weight, bias
        # NOTE: I don't know why training getting slower and slower if only `self.weight_mask` without `detach_()`
        self.module.weight = torch.mul(self.weight, _StraightThrough.apply(self.weight_score, self.weight_mask.detach_()))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias = torch.mul(self.bias, self.bias_mask)
        return self.module(*inputs)


class _StraightThrough(autograd.Function):
    """
    Straight through the gradient to the score, then the score = initial_score + sum(-lr * grad(weight) * weight).
    """
    @staticmethod
    def forward(self, score, masks):
        return masks

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class WeightScoreTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Collect all weight_score in wrappers as data used to calculate metrics.
    """
    def collect(self) -> Dict[str, Tensor]:
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = wrapper.weight_score.data
        return data


class MovementPruner(BasicPruner):
    r"""
    Movement pruner is an implementation of movement pruning.
    This is a "fine-pruning" algorithm, which means the masks may change during each fine-tuning step.
    Each weight element will be scored by the opposite of the sum of the product of weight and its gradient during each step.
    This means the weight elements moving towards zero will accumulate negative scores, the weight elements moving away from zero will accumulate positive scores.
    The weight elements with low scores will be masked during inference.

    The following figure from the paper shows the weight pruning by movement pruning.

    .. image:: ../../../img/movement_pruning.png
        :target: ../../../img/movement_pruning.png
        :alt:

    For more details, please refer to `Movement Pruning: Adaptive Sparsity by Fine-Tuning <https://arxiv.org/abs/2005.07683>`__.

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
    training_epochs : int
        The total epoch number for training the model.
        Make sure the total `optimizer.step()` in `training_epochs` is bigger than `cool_down_beginning_step`.
    warm_up_step : int
        The total `optimizer.step()` number before start pruning for warm up.
        Make sure `warm_up_step` is smaller than `cool_down_beginning_step`.
    cool_down_beginning_step: int
        The number of steps at which sparsity stops growing, note that the sparsity stop growing doesn't mean masks not changed.
        The sparsity after each `optimizer.step()` is:
        total_sparsity * (1 - (1 - (current_step - warm_up_step) / (cool_down_beginning_step - warm_up_step)) ** 3).

    Examples
    --------
        >>> import nni
        >>> from nni.compression.pytorch.pruning import MovementPruner
        >>> model = ...
        >>> # make sure you have used nni.trace to wrap the optimizer class before initialize
        >>> traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters())
        >>> trainer = ...
        >>> criterion = ...
        >>> config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
        >>> pruner = MovementPruner(model, config_list, trainer, traced_optimizer, criterion, 10, 3000, 27000)
        >>> masked_model, masks = pruner.compress()

    For detailed example please refer to :githublink:`examples/model_compress/pruning/movement_pruning_glue.py <examples/model_compress/pruning/movement_pruning_glue.py>`
    """
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Traceable, criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int, warm_up_step: int,
                 cool_down_beginning_step: int):
        self.trainer = trainer
        if isinstance(traced_optimizer, OptimizerConstructHelper):
            self.optimizer_helper = traced_optimizer
        else:
            self.optimizer_helper = OptimizerConstructHelper.from_trace(model, traced_optimizer)
        self.criterion = criterion
        self.training_epochs = training_epochs
        self.warm_up_step = warm_up_step
        self.cool_down_beginning_step = cool_down_beginning_step
        assert self.warm_up_step < self.cool_down_beginning_step, '`warm_up_step` should smaller than `cool_down_beginning_step`'
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    def cubic_schedule(self, current_step: int):
        if self.warm_up_step < current_step <= self.cool_down_beginning_step:
            wrapper_dict = self.get_modules_wrapper()
            for config in self.config_list:
                current_sparsity = config['total_sparsity'] * (1 - (1 - (current_step - self.warm_up_step) / (self.cool_down_beginning_step - self.warm_up_step)) ** 3)
                for op_name in config['op_names']:
                    wrapper_dict[op_name].config['total_sparsity'] = current_sparsity

    def reset_tools(self):
        if self.metrics_calculator is None:
            self.metrics_calculator = StraightMetricsCalculator()
        if self.sparsity_allocator is None:
            self.sparsity_allocator = NormalSparsityAllocator(self, continuous_mask=False)

        # use Adam to update the weight_score
        params = [{"params": [p for n, p in self.bound_model.named_parameters() if "weight_score" in n and p.requires_grad]}]
        optimizer = Adam(params, 1e-2)
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
            self.data_collector = WeightScoreTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper, self.criterion, self.training_epochs, opt_after_tasks=[_optimizer_patch])
        else:
            self.data_collector.reset()

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        Create a wrapper module to replace the original one.
        Different from the parent function, use `PrunerScoredModuleWrapper` instead of `PrunerModuleWrapper`.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerScoredModuleWrapper(layer.module, layer.name, config)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def compress(self) -> Tuple[Module, Dict]:
        # sparsity grow from 0
        for _, wrapper in self.get_modules_wrapper().items():
            wrapper.config['total_sparsity'] = 0
        result = super().compress()
        # del weight_score
        for _, wrapper in self.get_modules_wrapper().items():
            wrapper.weight_score = None
        return result
