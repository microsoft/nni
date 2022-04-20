# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gc
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module

_logger = logging.getLogger(__name__)


class Task:
    # NOTE: If we want to support multi-thread, this part need to refactor, maybe use file and lock to sync.
    _reference_counter = {}

    def __init__(self, task_id: int, model_path: str, masks_path: str, config_list_path: str,
                 speed_up: Optional[bool] = True, finetune: Optional[bool] = True, evaluate: Optional[bool] = True):
        """
        Parameters
        ----------
        task_id
            The unique id of task.
        model_path
            The path of the unwrapped pytorch model that will be pruned in this task.
        masks_path
            The path of the masks that applied on the model before pruning.
        config_list_path
            The path of the config list that used in this task.
        speed_up
            Control if this task needs speed up, True means use scheduler default value, False means no speed up.
        finetune
            Control if this task needs finetune, True means use scheduler default value, False means no finetune.
        evaluate
            Control if this task needs evaluate, True means use scheduler default value, False means no evaluate.
        """
        self.task_id = task_id
        self.model_path = model_path
        self.masks_path = masks_path
        self.config_list_path = config_list_path

        self.speed_up = speed_up
        self.finetune = finetune
        self.evaluate = evaluate

        self.status = 'Pending'
        self.score: Optional[float] = None

        self.state = {}

        for ref in self.referenced_paths():
            self._reference_counter.setdefault(ref, 0)
            self._reference_counter[ref] += 1

        self._cleaned = False

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'model_path': str(self.model_path),
            'masks_path': str(self.masks_path),
            'config_list_path': str(self.config_list_path),
            'speed_up': self.speed_up,
            'finetune': self.finetune,
            'evaluate': self.evaluate,
            'status': self.status,
            'score': self.score,
            'state': self.state
        }

    def load_data(self) -> Tuple[Module, Dict[str, Dict[str, Tensor]], List[Dict]]:
        """
        Returns
        -------
        Tuple[Module, Dict[str, Dict[str, Tensor]], List[Dict]]
            Return the model pruning in this task, the masks of the model before pruning,
            the config list used in this task.
        """
        model = torch.load(self.model_path)
        masks = torch.load(self.masks_path)
        with Path(self.config_list_path).open('r') as f:
            config_list = json_tricks.load(f)
        return model, masks, config_list

    def referenced_paths(self) -> List[str]:
        """
        Return the path list that need to count reference in this task.
        """
        return [self.model_path, self.masks_path, self.config_list_path]

    def clean_up(self):
        """
        Counter of referenced file paths subtract 1. If the counter reach 0, then delete the file.
        """
        if not self._cleaned:
            for ref in self.referenced_paths():
                self._reference_counter[ref] -= 1
                if self._reference_counter[ref] <= 0:
                    os.remove(ref)
                    if self._reference_counter[ref] < 0:
                        _logger.warning('Referance counter error, the number of %s is %d',
                                        ref, self._reference_counter[ref])
            self._cleaned = True
        else:
            _logger.warning('Already clean up task %d', self.task_id)


class TaskResult:
    def __init__(self, task_id: int, compact_model: Module, compact_model_masks: Dict[str, Dict[str, Tensor]],
                 pruner_generated_masks: Dict[str, Dict[str, Tensor]], score: Optional[float]) -> None:
        """
        Parameters
        ----------
        task_id
            The unique id of task.
        compact_model
            The unwrapped compact pytorch model after pruning. If the compact model has been speeduped during the pruning process,
            it will have a smaller structure compare with the model before pruning.
            If the compact model has not been speeduped, it will have the same structure with the model before pruning.
        compact_model_masks
            The masks on the compact model. If the compact model has been speeduped during the pruning process,
            the `compact_model_masks` is always an empty dict. If the compact model has not been speeduped,
            the `compact_model_masks` is same as `pruner_generated_masks`.
        pruner_generated_masks
            The masks that can apply on the before pruning model. It is always the output of `pruner.compress()`.
            TODO: If the compact model has been speeduped, the auto infer masks maybe also need.
        score
            The score of the pruning effect. i.e., the accuracy or latency after pruning.
        """
        self.task_id = task_id
        self.compact_model = compact_model
        self.compact_model_masks = compact_model_masks
        self.pruner_generated_masks = pruner_generated_masks
        self.score = score


class BasePruningScheduler:
    def generate_task(self) -> Optional[Task]:
        """
        Returns
        -------
        Optional[Task]
            Return the next pruning task.
        """
        raise NotImplementedError()

    def record_task_result(self, task_result: TaskResult):
        """
        Parameters
        ----------
        task_result
            The result of the task
        """
        raise NotImplementedError()

    def pruning_one_step(self, task: Task) -> TaskResult:
        """
        Pruning the model defined in task.

        Parameters
        ----------
        task
            The pruning task in this step.

        Returns
        -------
        TaskResult
            Return the result of the task in this step.
        """
        raise NotImplementedError()

    def get_best_result(self) -> Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]:
        """
        Returns
        -------
        Tuple[int, Module, Dict[str, Dict[str, Tensor]], float, List[Dict]]
            Return the task result that has the best performance,
            inculde task id, the compact model, the masks on the compact model, score and config list used in this task.
        """
        raise NotImplementedError()

    def compress(self):
        """
        The pruning schedule main loop.
        """
        task = self.generate_task()

        while task is not None:
            task_result = self.pruning_one_step(task)
            self.record_task_result(task_result)
            del task_result
            gc.collect()
            task = self.generate_task()
