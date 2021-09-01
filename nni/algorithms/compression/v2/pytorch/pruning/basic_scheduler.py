# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Pruner, BasePruningScheduler
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
        """
        self.pruner = pruner
        self.task_generator = task_generator
        self.finetuner = finetuner
        self.speed_up = speed_up
        self.dummy_input = dummy_input
        self.evaluator = evaluator

    def generate_task(self) -> Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        return self.task_generator.next()

    def record_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]], score: float,
                           up_model_masks: Dict[str, Dict[str, Tensor]]):
        self.task_generator.receive_task_result(task_id, pruned_model, masks, score, up_model_masks)

    def pruning_one_step(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]]) \
            -> Tuple[Module, Dict[str, Dict[str, Tensor]], float, Dict[str, Dict[str, Tensor]]]:
        # pruning model
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)
        model, masks = self.pruner.compress()
        up_model_masks = deepcopy(masks)

        # show the pruning effect
        self.pruner.show_pruned_weights()
        self.pruner._unwrap_model()

        # speed up
        # TODO: speed up only support mask file path as input, maybe we need also support masks directly.
        if self.speed_up:
            tmp_masks = {}
            for name, mask in masks.items():
                tmp_masks[name] = {}
                tmp_masks[name]['weight'] = mask.get('weight_mask')
                if 'bias' in masks:
                    tmp_masks[name]['bias'] = mask.get('bias_mask')
            torch.save(tmp_masks, Path('./temp_masks.pth'))
            ModelSpeedup(model, self.dummy_input, Path('./temp_masks.pth')).speedup_model()
            os.remove('./temp_masks.pth')
            masks = {}

        # finetune
        if self.finetuner is not None:
            if self.speed_up:
                self.finetuner(model)
            else:
                self.pruner._wrap_model()
                self.finetuner(model)
                self.pruner._unwrap_model()

        # evaluate
        score = self.evaluator(model) if self.evaluator is not None else None

        return model, masks, score, up_model_masks

    def get_best_result(self) -> Optional[Tuple[int, Module, Dict[str, Dict[str, Tensor]], float]]:
        return self.task_generator.get_best_result()
