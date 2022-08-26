# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, List, Tuple, Callable, Optional, Union, overload

import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Pruner, BasePruningScheduler, Task, TaskResult
from nni.compression.pytorch.speedup import ModelSpeedup

from .tools import TaskGenerator
from ..utils import Evaluator, LightningEvaluator, TorchEvaluator

_logger = logging.getLogger(__name__)

_LEGACY_FINETUNER = Callable[[Module], None]
_LEGACY_EVALUATOR = Callable[[Module], float]


# TODO: remove in nni v3.0.
class EvaluatorBasedPruningScheduler(BasePruningScheduler):
    evaluator: LightningEvaluator | TorchEvaluator
    using_evaluator: bool
    finetuner: _LEGACY_FINETUNER
    _evaluator: _LEGACY_EVALUATOR
    dummy_input: Any

    def _init_evaluator(self, model: Module, new_api: List[str], new_init_kwargs: Dict, old_api: List[str],
                        old_init_kwargs: Dict, args: Tuple, kwargs: Dict) -> Dict:
        # for fake __init__ overload, parsing args and kwargs,
        # initializing evaluator or [finetuner, evaluator, dummy_input], return the remaining arguments.
        if (len(args) > 0 and isinstance(args[0], Evaluator)) or \
                (len(args) == 0 and isinstance(kwargs.get('evaluator', None), Evaluator)):
            init_kwargs = self._parse_args(new_api, args, kwargs, new_init_kwargs)
            self.evaluator: LightningEvaluator | TorchEvaluator = init_kwargs.pop('evaluator')
            if not self.evaluator._initialization_complete:
                self.evaluator._init_optimizer_helpers(model)  # type: ignore
            self.using_evaluator = True
        else:
            init_kwargs = self._parse_args(old_api, args, kwargs, old_init_kwargs)
            self.finetuner: _LEGACY_FINETUNER = init_kwargs.pop('finetuner')
            self._evaluator: _LEGACY_EVALUATOR = init_kwargs.pop('evaluator')
            self.dummy_input = init_kwargs.pop('dummy_input')
            self.using_evaluator = False
            warn_msg = f'The old API ...{",".join(old_api)} will be deprecated after NNI v3.0,' + \
                       f'please using the new one ...{",".join(new_api)}'
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


class PruningScheduler(EvaluatorBasedPruningScheduler):
    """
    Parameters
    ----------
    pruner
        The pruner used in pruner scheduler.
        The scheduler will use `Pruner.reset(model, config_list)` to reset it in each iteration.
    task_generator
        Used to generate task for each iteration.
    finetuner
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration if reset_weight is False,
        will be called at the beginning of each iteration otherwise.
    speedup
        If set True, speedup the model at the end of each iteration to make the pruned model compact.
    dummy_input
        If `speedup` is True, `dummy_input` is required for tracing the model in speedup.
    evaluator
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    reset_weight
        If set True, the model weight will reset to the origin model weight at the end of each iteration step.
    """

    @overload
    def __init__(self, pruner: Pruner, task_generator: TaskGenerator, evaluator: LightningEvaluator | TorchEvaluator,
                 speedup: bool = False, reset_weight: bool = False):
        ...

    @overload
    def __init__(self, pruner: Pruner, task_generator: TaskGenerator, finetuner: _LEGACY_FINETUNER | None = None,
                 speedup: bool = False, dummy_input: Optional[Tensor] = None, evaluator: _LEGACY_EVALUATOR | None = None,
                 reset_weight: bool = False):
        ...

    def __init__(self, pruner: Pruner, task_generator: TaskGenerator, *args, **kwargs) -> None:
        # TODO: remove in nni v3.0. Fake overload.
        new_api = ['evaluator', 'speedup', 'reset_weight']
        new_init_kwargs = {'evaluator': None, 'speedup': False, 'reset_weight': False}
        old_api = ['finetuner', 'speedup', 'dummy_input', 'evaluator', 'reset_weight']
        old_init_kwargs = {'finetuner': None, 'evaluator': None, 'dummy_input': None, 'speedup': False, 'reset_weight': False}
        init_kwargs = self._init_evaluator(None, new_api, new_init_kwargs, old_api, old_init_kwargs, args, kwargs)  # type: ignore

        self.pruner = pruner
        self.task_generator = task_generator
        self.speedup = init_kwargs['speedup']
        self.reset_weight = init_kwargs['reset_weight']

    def reset(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.task_generator.reset(model, config_list, masks)

    def generate_task(self) -> Optional[Task]:
        return self.task_generator.next()

    def record_task_result(self, task_result: TaskResult):
        self.task_generator.receive_task_result(task_result)

    def pruning_one_step_normal(self, task: Task) -> TaskResult:
        """
        generate masks -> speedup -> finetune -> evaluate
        """
        model, masks, config_list = task.load_data()

        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)

        # pruning model
        compact_model, pruner_generated_masks = self.pruner.compress()
        compact_model_masks = deepcopy(pruner_generated_masks)

        # show the pruning effect
        self.pruner.show_pruned_weights()
        self.pruner._unwrap_model()

        # speedup
        if self.speedup and task.speedup:
            if self.using_evaluator:
                ModelSpeedup(compact_model, self.evaluator.get_dummy_input(), pruner_generated_masks).speedup_model()
                compact_model_masks = {}
            else:
                ModelSpeedup(compact_model, self.dummy_input, pruner_generated_masks).speedup_model()
                compact_model_masks = {}

        # finetune
        if self.using_evaluator:
            if task.finetune:
                self.evaluator.bind_model(compact_model)  # type: ignore
                if self.speedup:
                    self.evaluator.finetune()
                else:
                    self.pruner._wrap_model()
                    self.evaluator.finetune()
                    self.pruner._unwrap_model()
                self.evaluator.unbind_model()
        else:
            if self.finetuner is not None and task.finetune:
                if self.speedup:
                    self.finetuner(compact_model)
                else:
                    self.pruner._wrap_model()
                    self.finetuner(compact_model)
                    self.pruner._unwrap_model()

        # evaluate
        if self.using_evaluator:
            if task.evaluate:
                self.evaluator.bind_model(compact_model)  # type: ignore
                # TODO: support saving customized score
                if self.speedup:
                    score = self.evaluator.evaluate()
                else:
                    self.pruner._wrap_model()
                    score = self.evaluator.evaluate()
                    self.pruner._unwrap_model()
                score = score[0] if isinstance(score, tuple) else score
                self.evaluator.unbind_model()
            else:
                score = None
        else:
            if self._evaluator is not None and task.evaluate:
                if self.speedup:
                    score = self._evaluator(compact_model)  # type: ignore
                else:
                    self.pruner._wrap_model()
                    score = self._evaluator(compact_model)  # type: ignore
                    self.pruner._unwrap_model()
            else:
                score = None

        # clear model references
        self.pruner.clear_model_references()

        return TaskResult(task.task_id, compact_model, compact_model_masks, pruner_generated_masks, score)

    def pruning_one_step_reset_weight(self, task: Task) -> TaskResult:
        """
        finetune -> generate masks -> reset weight -> speedup -> evaluate
        """
        model, masks, config_list = task.load_data()

        checkpoint = deepcopy(model.state_dict())
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)

        # finetune
        if self.using_evaluator:
            if task.finetune:
                self.evaluator.bind_model(model)  # type: ignore
                self.evaluator.finetune()
                self.evaluator.unbind_model()
        else:
            if self.finetuner is not None and task.finetune:
                self.finetuner(model)

        # pruning model
        compact_model, pruner_generated_masks = self.pruner.compress()
        compact_model_masks = deepcopy(pruner_generated_masks)

        # show the pruning effect
        self.pruner.show_pruned_weights()
        self.pruner._unwrap_model()

        # reset model weight
        compact_model.load_state_dict(checkpoint)

        # speedup
        if self.speedup and task.speedup:
            if self.using_evaluator:
                ModelSpeedup(compact_model, self.evaluator.get_dummy_input(), pruner_generated_masks).speedup_model()
                compact_model_masks = {}
            else:
                ModelSpeedup(compact_model, self.dummy_input, pruner_generated_masks).speedup_model()
                compact_model_masks = {}

        # evaluate
        if self.using_evaluator:
            if task.evaluate:
                self.evaluator.bind_model(compact_model)  # type: ignore
                # TODO: support saving customized score
                if self.speedup:
                    score = self.evaluator.evaluate()
                else:
                    self.pruner._wrap_model()
                    score = self.evaluator.evaluate()
                    self.pruner._unwrap_model()
                score = score[0] if isinstance(score, tuple) else score
                self.evaluator.unbind_model()
            else:
                score = None
        else:
            if self._evaluator is not None and task.evaluate:
                if self.speedup:
                    score = self._evaluator(compact_model)  # type: ignore
                else:
                    self.pruner._wrap_model()
                    score = self._evaluator(compact_model)  # type: ignore
                    self.pruner._unwrap_model()
            else:
                score = None

        # clear model references
        self.pruner.clear_model_references()

        return TaskResult(task.task_id, compact_model, compact_model_masks, pruner_generated_masks, score)

    def pruning_one_step(self, task: Task) -> TaskResult:
        if self.reset_weight:
            result = self.pruning_one_step_reset_weight(task)
        else:
            result = self.pruning_one_step_normal(task)
        torch.cuda.empty_cache()
        return result

    def get_best_result(self) -> Optional[Tuple[Union[int, str], Module, Dict[str, Dict[str, Tensor]], Optional[float], List[Dict]]]:
        return self.task_generator.get_best_result()
