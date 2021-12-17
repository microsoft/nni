# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Pruner, BasePruningScheduler, Task, TaskResult
from nni.compression.pytorch.speedup import ModelSpeedup

from .tools import TaskGenerator


class PruningScheduler(BasePruningScheduler):
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
        It will be called at the end of each iteration if reset_weight is False, will be called at the beginning of each iteration otherwise.
    speed_up
        If set True, speed up the model at the end of each iteration to make the pruned model compact.
    dummy_input
        If `speed_up` is True, `dummy_input` is required for tracing the model in speed up.
    evaluator
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    reset_weight
        If set True, the model weight will reset to the origin model weight at the end of each iteration step.
    """
    def __init__(self, pruner: Pruner, task_generator: TaskGenerator, finetuner: Callable[[Module], None] = None,
                 speed_up: bool = False, dummy_input: Tensor = None, evaluator: Optional[Callable[[Module], float]] = None,
                 reset_weight: bool = False):
        self.pruner = pruner
        self.task_generator = task_generator
        self.finetuner = finetuner
        self.speed_up = speed_up
        self.dummy_input = dummy_input
        self.evaluator = evaluator
        self.reset_weight = reset_weight

    def reset(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.task_generator.reset(model, config_list, masks)

    def generate_task(self) -> Optional[Task]:
        return self.task_generator.next()

    def record_task_result(self, task_result: TaskResult):
        self.task_generator.receive_task_result(task_result)

    def pruning_one_step_normal(self, task: Task) -> TaskResult:
        """
        generate masks -> speed up -> finetune -> evaluate
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

        # speed up
        if self.speed_up and task.speed_up:
            ModelSpeedup(compact_model, self.dummy_input, pruner_generated_masks).speedup_model()
            compact_model_masks = {}

        # finetune
        if self.finetuner is not None and task.finetune:
            if self.speed_up:
                self.finetuner(compact_model)
            else:
                self.pruner._wrap_model()
                self.finetuner(compact_model)
                self.pruner._unwrap_model()

        # evaluate
        if self.evaluator is not None and task.evaluate:
            if self.speed_up:
                score = self.evaluator(compact_model)
            else:
                self.pruner._wrap_model()
                score = self.evaluator(compact_model)
                self.pruner._unwrap_model()
        else:
            score = None

        # clear model references
        self.pruner.clear_model_references()

        return TaskResult(task.task_id, compact_model, compact_model_masks, pruner_generated_masks, score)

    def pruning_one_step_reset_weight(self, task: Task) -> TaskResult:
        """
        finetune -> generate masks -> reset weight -> speed up -> evaluate
        """
        model, masks, config_list = task.load_data()
        checkpoint = deepcopy(model.state_dict())
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)

        # finetune
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

        # speed up
        if self.speed_up and task.speed_up:
            ModelSpeedup(compact_model, self.dummy_input, pruner_generated_masks).speedup_model()
            compact_model_masks = {}

        # evaluate
        if self.evaluator is not None and task.evaluate:
            if self.speed_up:
                score = self.evaluator(compact_model)
            else:
                self.pruner._wrap_model()
                score = self.evaluator(compact_model)
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

    def get_best_result(self) -> Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]]:
        return self.task_generator.get_best_result()
