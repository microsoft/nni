# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Pruner, BasePruningScheduler, Task, TaskResult
from nni.compression.pytorch.speedup import ModelSpeedup

from .tools import TaskGenerator


class PruningScheduler(BasePruningScheduler):
    def __init__(self, pruner: Pruner, task_generator: TaskGenerator, finetuner: Callable[[Module], None] = None,
                 speed_up: bool = False, dummy_input: Tensor = None, evaluator: Optional[Callable[[Module], float]] = None):
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
        speed_up
            If set True, speed up the model in each iteration.
        dummy_input
            If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
        evaluator
            Evaluate the pruned model and give a score.
            If evaluator is None, the best result refers to the latest result.
        """
        self.pruner = pruner
        self.task_generator = task_generator
        self.finetuner = finetuner
        self.speed_up = speed_up
        self.dummy_input = dummy_input
        self.evaluator = evaluator

    def generate_task(self) -> Optional[Task]:
        return self.task_generator.next()

    def record_task_result(self, task_result: TaskResult):
        self.task_generator.receive_task_result(task_result)

    def pruning_one_step(self, task: Task) -> TaskResult:
        model, masks, config_list = task.load_data()

        # pruning model
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)
        compact_model, old_structure_masks = self.pruner.compress()
        compact_model_masks = deepcopy(old_structure_masks)

        # show the pruning effect
        self.pruner.show_pruned_weights()
        self.pruner._unwrap_model()

        # speed up
        # TODO: speed up only support mask file path as input, maybe we need also support masks directly.
        if self.speed_up:
            torch.save(old_structure_masks, Path('./temp_masks.pth'))
            ModelSpeedup(compact_model, self.dummy_input, Path('./temp_masks.pth')).speedup_model()
            os.remove('./temp_masks.pth')
            compact_model_masks = {}

        # finetune
        if self.finetuner is not None:
            if self.speed_up:
                self.finetuner(compact_model)
            else:
                self.pruner._wrap_model()
                self.finetuner(compact_model)
                self.pruner._unwrap_model()

        # evaluate
        score = self.evaluator(compact_model) if self.evaluator is not None else None

        # clear model references
        self.pruner.clear_model_references()

        return TaskResult(task.task_id, compact_model, compact_model_masks, old_structure_masks, score)

    def get_best_result(self) -> Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]]:
        return self.task_generator.get_best_result()
