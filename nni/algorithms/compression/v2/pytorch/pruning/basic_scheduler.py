# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Pruner
from nni.compression.pytorch.speedup import ModelSpeedup

from .tools import TaskGenerator

_logger = logging.getLogger(__name__)


class ToolBasedPruningScheduler:
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

        self.latest_model = None
        self.latest_masks = None

    def compress_one_step(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]]) -> Tuple[Module, List[Dict], Optional[float]]:
        # compress model
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)
        model, masks = self.pruner.compress()

        # show the pruning effect
        self.pruner.show_pruned_weights()

        # apply masks to sparsify model
        self.pruner._unwrap_model()

        # speed up
        # TODO: speed up only support mask file path as input, maybe we need also support masks directly.
        tmp_masks = {}
        for name, mask in masks.items():
            tmp_masks[name] = {}
            tmp_masks[name]['weight'] = mask.get('weight_mask')
            if 'bias' in masks:
                tmp_masks[name]['bias'] = mask.get('bias_mask')
        torch.save(tmp_masks, Path('./temp_masks.pth'))
        if self.speed_up:
            ModelSpeedup(model, self.dummy_input, Path('./temp_masks.pth')).speedup_model()
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

        return model, masks, score

    def compress(self) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        """
        Pruning the model step by step, until task generator return config list which is None.
        Returns
        -------
        Tuple[Module, Dict[str, Dict[str, Tensor]]]
            Return the pruned model and the masks on the pruned model returned by the last compress step.
        """
        iteration = 0
        task_id, model, config_list, masks = self.task_generator.next()

        while task_id is not None:
            self.latest_model, self.latest_masks, score = self.compress_one_step(model, config_list, masks)
            _logger.info('\nIteration %d\ntask id: %d\nscore: %s\nconfig list:\n%s', iteration, task_id, str(score),
                         json_tricks.dumps(config_list, indent=4))

            self.task_generator.receive_task_result(task_id, self.latest_model, self.latest_masks, score)
            task_id, model, config_list, masks = self.task_generator.next()

            iteration += 1

        return self.latest_model, self.latest_masks

    def get_best_result(self) -> Optional[Tuple[Module, Dict[str, Dict[str, Tensor]]]]:
        return self.task_generator.get_best_result()
