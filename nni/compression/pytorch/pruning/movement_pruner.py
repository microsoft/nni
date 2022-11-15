# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Dict, List, Tuple, Callable, overload
from typing_extensions import Literal

import torch
from torch import autograd, Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer, Adam

from nni.compression.pytorch.base import PrunerModuleWrapper, LayerInfo
from nni.compression.pytorch.pruning.basic_pruner import EvaluatorBasedPruner, NORMAL_SCHEMA, EXCLUDE_SCHEMA, INTERNAL_SCHEMA
from nni.compression.pytorch.utils import CompressorSchema

from .tools.base import EvaluatorBasedDataCollector, TrainerBasedDataCollector

from .tools import (
    NormalSparsityAllocator,
    ThresholdSparsityAllocator,
    StraightMetricsCalculator
)

from ..utils import (
    Evaluator,
    Scaling
)

from ..utils.docstring import _EVALUATOR_DOCSTRING
from ..utils.external.huggingface import parser_factory

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
    def __init__(self, module: Module, module_name: str, config: Dict, score_size: List[int] | None = None):
        super().__init__(module, module_name, config)
        self.weight_score = Parameter(torch.empty(score_size)) \
            if score_size is not None else Parameter(torch.empty_like(module.weight))  # type: ignore
        torch.nn.init.constant_(self.weight_score, val=0.0)

    def forward(self, *inputs):
        repeat = [a // b for a, b in zip(self.weight.shape, self.weight_score.shape)]  # type: ignore
        weight_score = self.weight_score
        for dim, num in enumerate(repeat):
            weight_score = weight_score.repeat_interleave(num, dim=dim)
        self.module.weight = torch.mul(self.weight, _StraightThrough.apply(weight_score, self.weight_mask))  # type: ignore
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias = torch.mul(self.bias, self.bias_mask)  # type: ignore
        return self.module(*inputs)


class _StraightThrough(autograd.Function):
    """
    Straight through the gradient to the score, then the score = initial_score + sum(-lr * grad(weight) * weight).
    """
    @staticmethod
    def forward(ctx, score, masks):
        return masks

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class WeightScoreTrainerBasedDataCollector(TrainerBasedDataCollector):
    """
    Collect all weight_score in wrappers as data used to calculate metrics.
    """
    def collect(self) -> Dict[str, Tensor]:
        assert self.compressor.bound_model is not None
        for _ in range(self.training_epochs):
            self.trainer(self.compressor.bound_model, self.optimizer, self.criterion)

        data = {}
        target_name = 'weight'
        for _, wrapper in self.compressor.get_modules_wrapper().items():
            data[wrapper.name] = {target_name: wrapper.weight_score.data}  # type: ignore
        return data


class EvaluatorBasedScoreDataCollector(EvaluatorBasedDataCollector):
    """
    Collect all weight_score in wrappers as data used to calculate metrics.
    """
    def collect(self) -> Dict[str, Tensor]:
        assert self.compressor.bound_model is not None
        self.evaluator.train(max_steps=self.max_steps, max_epochs=self.max_epochs)

        data = {}
        target_name = 'weight'
        for module_name, wrapper in self.compressor.get_modules_wrapper().items():
            target_score: Tensor = getattr(wrapper, f'{target_name}_score')
            data[module_name] = {target_name: target_score.data.clone()}
        return data


class MovementPruner(EvaluatorBasedPruner):
    __doc__ = r"""
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

    evaluator
        ``evaluator`` is used to replace the previous ``trainer``, ``traced_optimizer`` and ``criterion`` API.
        {evaluator_docstring}
        The old API (``trainer``, ``traced_optimizer`` and ``criterion``) is still supported and will be deprecated in v3.0.
        If you want to consult the old API, please refer to `v2.8 pruner API <https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html>`__.
    warm_up_step
        The total `optimizer.step()` number before start pruning for warm up.
        Make sure ``warm_up_step`` is smaller than ``cool_down_beginning_step``.
    cool_down_beginning_step
        The number of steps at which sparsity stops growing, note that the sparsity stop growing doesn't mean masks not changed.
        The sparsity after each `optimizer.step()` is:
        total_sparsity * (1 - (1 - (current_step - warm_up_step) / (cool_down_beginning_step - warm_up_step)) ** 3).
    training_epochs
        The total epoch number for training the model.
        Make sure the total `optimizer.step()` in ``training_epochs`` is bigger than `cool_down_beginning_step`.
        If both ``training_epochs`` and ``training_steps`` are set, pruning will stop when either is reached.
    training_steps
        The total step number for training the model.
        Make sure ``training_epochs`` is bigger than ``cool_down_beginning_step``.
        If both ``training_epochs`` and ``training_steps`` are set, pruning will stop when either is reached.
    regular_scale
        Use to scale the movement score regular loss. In 'soft' mode, higher regular scale means higher final sparsity.
        The recommended range is 1 ~ 30.
    movement_mode
        'hard' or 'soft'. Note that in 'soft' mode, ``sparsity`` set in the ``config_list`` means the sparsify threshold,
        'soft' mode cannot precisely control the sparsity rate, but usually has higher performance compared with 'hard' mode.
        ``sparsity`` in 'soft' mode usually set to ``0.1``, and using ``regular_scale`` to control the final relative sparsity.

        For detailed differences between 'hard' and 'soft', please refer to the paper.
        In short, 'hard' means that the corresponding layer is pruned to a fixed ratio by the topk method according to the movement score,
        which is the sparsity ratio set in config_list.
        'soft' means that the final sparsity size will not be fixed, but the generation of the mask will be controlled by a threshold,
        and the positions corresponding to scores below the threshold will be masked during the movement training process.
    sparse_granularity
        This is an experimental interface, by default, apply 'finegrained' pruning. If 'auto' is set, will try to apply structure pruning.
        For the attention layer, will apply block sparse with size [head_width, head_width]. For the following two linear layers (FFN),
        will apply output channel pruning for the first linear, and the input channel pruning for the second one.
        'auto' only support partial hugingface transformers right now (bart, bert, t5).

    Notes
    -----
    For detailed example please refer to :githublink:`examples/model_compress/pruning/movement_pruning_glue.py <examples/model_compress/pruning/movement_pruning_glue.py>`
    """.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: Module, config_list: List[Dict], evaluator: Evaluator, warm_up_step: int,
                 cool_down_beginning_step: int, training_epochs: int | None = None, training_steps: int | None = None,
                 regular_scale: float | None = None, movement_mode: Literal['hard', 'soft'] = 'hard',
                 sparse_granularity: Literal['auto', 'finegrained'] = 'finegrained'):
        ...

    @overload
    def __init__(self, model: Module, config_list: List[Dict], trainer: Callable[[Module, Optimizer, Callable], None],
                 traced_optimizer: Optimizer, criterion: Callable[[Tensor, Tensor], Tensor], training_epochs: int, warm_up_step: int,
                 cool_down_beginning_step: int):
        ...

    def __init__(self, model: Module, config_list: List[Dict], *args, **kwargs):
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'warm_up_step', 'cool_down_beginning_step', 'training_epochs', 'training_steps', 'regular_scale',
                   'movement_mode', 'sparse_granularity']
        old_api = ['trainer', 'traced_optimizer', 'criterion', 'training_epochs', 'warm_up_step', 'cool_down_beginning_step']
        init_kwargs = {'training_epochs': None, 'training_steps': None, 'regular_scale': None, 'movement_mode': 'hard',
                       'sparse_granularity': 'finegrained'}
        init_kwargs = self._init_evaluator(model, new_api, old_api, init_kwargs, args, kwargs)

        self.training_epochs: int = init_kwargs['training_epochs']
        self.training_steps: int | None = init_kwargs['training_steps'] if self.using_evaluator else None
        self.warm_up_step: int = init_kwargs['warm_up_step']
        self.cool_down_beginning_step: int = init_kwargs['cool_down_beginning_step']
        self.regular_scale: int | None = init_kwargs['regular_scale'] if self.using_evaluator else None
        self.movement_mode: Literal['hard', 'soft'] | None = init_kwargs['movement_mode'] if self.using_evaluator else None
        self.sparse_granularity = init_kwargs['sparse_granularity'] if self.using_evaluator else None
        assert self.warm_up_step < self.cool_down_beginning_step, '`warm_up_step` should smaller than `cool_down_beginning_step`'

        self._model_parser = parser_factory(model)
        super().__init__(model, config_list)

    def _validate_config_before_canonical(self, model: Module, config_list: List[Dict]):
        schema_list = [deepcopy(NORMAL_SCHEMA), deepcopy(EXCLUDE_SCHEMA), deepcopy(INTERNAL_SCHEMA)]
        schema = CompressorSchema(schema_list, model, _logger)
        schema.validate(config_list)

    def cubic_schedule(self, current_step: int):
        wrapper_dict = self.get_modules_wrapper()
        for config in self.config_list:
            current_sparsity = config['total_sparsity'] * self._cubic_scale(current_step)
            for op_name in config['op_names']:
                # There is an unreachable pyright error if `wrapper_dict[op_name].config['total_sparsity'] = current_sparsity`,
                # seems a pyright bug...
                wrapper_config = wrapper_dict[op_name].config
                wrapper_config['total_sparsity'] = current_sparsity

    def _cubic_scale(self, current_step: int):
        if self.warm_up_step > current_step:
            return 0
        elif current_step > self.cool_down_beginning_step:
            return 1
        else:
            return 1 - (1 - (current_step - self.warm_up_step) / (self.cool_down_beginning_step - self.warm_up_step)) ** 3

    def _create_scalers(self) -> Scaling | Dict[str, Dict[str, Scaling]]:
        assert self.bound_model is not None
        if self.sparse_granularity and self.sparse_granularity == 'auto' and self._model_parser:
            scalers = {}
            for module_name, wrapper in self.get_modules_wrapper().items():
                if self._model_parser.is_attention(module_name):
                    num_heads = self._model_parser.get_num_heads(module_name, self.bound_model)
                    if num_heads <= 0:
                        scalers[module_name] = {'_default': Scaling([1])}
                    else:
                        # assume attention layer weights are 2D
                        weight_h: int = wrapper.module.weight.shape[0]  # type: ignore
                        weight_w: int = wrapper.module.weight.shape[1]  # type: ignore
                        if weight_h % num_heads != 0 or weight_w % num_heads != 0:
                            scalers[module_name] = {'_default': Scaling([1])}
                        else:
                            block_h = weight_h // num_heads
                            block_w = weight_w // num_heads
                            scalers[module_name] = {'_default': Scaling([block_h, block_w])}
                elif self._model_parser.is_ffn(module_name, ffn_num=1):
                    scalers[module_name] = {'_default': Scaling([1, wrapper.module.weight.shape[1]])}  # type: ignore
                elif self._model_parser.is_ffn(module_name, ffn_num=2):
                    scalers[module_name] = {'_default': Scaling([wrapper.module.weight.shape[0], 1])}  # type: ignore
                else:
                    scalers[module_name] = {'_default': Scaling([1])}
        else:
            scalers = Scaling([1])
        return scalers

    def reset_tools(self):
        scalers = self._create_scalers()
        if not hasattr(self, 'metrics_calculator'):
            self.metrics_calculator = StraightMetricsCalculator()
        if not hasattr(self, 'sparsity_allocator'):
            if self.movement_mode == 'soft':
                self.sparsity_allocator = ThresholdSparsityAllocator(self, scalers=scalers, continuous_mask=False)
            else:
                self.sparsity_allocator = NormalSparsityAllocator(self, scalers=scalers, continuous_mask=False)

        # use Adam to update the weight_score
        assert self.bound_model is not None
        params = [{"params": [p for n, p in self.bound_model.named_parameters() if "weight_score" in n and p.requires_grad]}]
        optimizer = Adam(params, 1e-2)
        self.step_counter = 0

        # TODO: waiting for api stable and experiemnts to prove this scheduler is needed.
        # def lr_lambda(current_step: int):
        #     if current_step < self.warm_up_step:
        #         return float(current_step) / self.warm_up_step
        #     return max(0.0, float(147264 - current_step) / float(147264 - self.warm_up_step))

        # lr_scheduler = LambdaLR(optimizer, lr_lambda)

        # update the masks after each optimzier step
        def _optimizer_patch():
            optimizer.step()
            optimizer.zero_grad()
            self.step_counter += 1
            if self.step_counter > self.warm_up_step:
                self.cubic_schedule(self.step_counter)
                data = {}
                target_name = 'weight'
                for wrapper in self.get_modules_wrapper().values():
                    data[wrapper.name] = {target_name: wrapper.weight_score.data}
                metrics = self.metrics_calculator.calculate_metrics(data)  # type: ignore
                masks = self.sparsity_allocator.generate_sparsity(metrics)  # type: ignore
                self.load_masks(masks)

        def _loss_patch(origin_loss: Tensor):
            if self.regular_scale is not None:
                l1_reg = 0
                count = 0
                for wrapper in self.get_modules_wrapper().values():
                    l1_reg += torch.norm(torch.sigmoid(wrapper.weight_score), p=1) / wrapper.weight_score.numel()  # type: ignore
                    count += 1
                return origin_loss + self.regular_scale * self._cubic_scale(self.step_counter) * l1_reg / count
            else:
                return origin_loss

        if self.using_evaluator:
            # TODO: move to other place in nni v3.0
            self.evaluator.unbind_model()
            self.evaluator.bind_model(self.bound_model, self.get_origin2wrapped_parameter_name_map())  # type: ignore
            if not hasattr(self, 'data_collector'):
                self.data_collector = EvaluatorBasedScoreDataCollector(self, self.evaluator,
                                                                       after_opt_step_tasks=[_optimizer_patch],
                                                                       max_epochs=self.training_epochs,
                                                                       max_steps=self.training_steps,
                                                                       loss_patch=_loss_patch)
            else:
                self.data_collector.reset(after_opt_step_tasks=[_optimizer_patch])
        else:
            if not hasattr(self, 'data_collector'):
                self.data_collector = WeightScoreTrainerBasedDataCollector(self, self.trainer, self.optimizer_helper,
                                                                           self.criterion, self.training_epochs,
                                                                           opt_after_tasks=[_optimizer_patch])
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
        assert self.bound_model is not None
        # TODO: merge with _create_scalers after nni v3.0
        if self.sparse_granularity and self.sparse_granularity == 'auto' and self._model_parser:
            if self._model_parser.is_attention(layer.name):
                num_heads = self._model_parser.get_num_heads(layer.name, self.bound_model)
                if num_heads <= 0:
                    score_size = None
                else:
                    if layer.module.weight.shape[0] % num_heads != 0 or layer.module.weight.shape[1] % num_heads != 0:  # type: ignore
                        score_size = None
                    else:
                        score_size = [num_heads, num_heads]
            elif self._model_parser.is_ffn(layer.name, ffn_num=1):
                score_size = [layer.module.weight.shape[0], 1]  # type: ignore
            elif self._model_parser.is_ffn(layer.name, ffn_num=2):
                score_size = [1, layer.module.weight.shape[1]]  # type: ignore
            else:
                score_size = None
        else:
            score_size = None
        wrapper = PrunerScoredModuleWrapper(layer.module, layer.name, config, score_size)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)  # type: ignore
        return wrapper

    def compress(self) -> Tuple[Module, Dict]:
        # sparsity grow from 0
        for wrapper in self.get_modules_wrapper().values():
            wrapper.config['total_sparsity'] = 0
        result = super().compress()
        if self.using_evaluator:
            self.evaluator.unbind_model()
        return result
