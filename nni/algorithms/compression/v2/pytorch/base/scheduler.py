# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

import json_tricks
import torch
from torch.nn import Module
from torch.tensor import Tensor

from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.v2.pytorch.utils import apply_compression_results
from .pruner import Pruner

_logger = logging.getLogger(__name__)


CONFIG_LIST_NAME = 'config_list.json'
MODEL_NAME = 'pruned_model.pth'
MASKS_NAME = 'masks.pth'
PRE_TASK_ID = 'preTaskId'
SCORE = 'score'
LOG_DIR = 'logDir'
STATUS = 'status'


@dataclass
class Task:
    """
    Task saves the related information about the task.
    """
    task_id: int
    pre_task_id: Optional[int]
    config_list: dict
    log_dir: Path
    score: Optional[float] = None
    status: dict = field(default_factory=dict)


class TaskGenerator:
    """
    This class used to generate config list for pruner in each iteration.
    """
    def __init__(self, origin_model: Module, origin_config_list: List[Dict] = [],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.'):
        assert isinstance(origin_model, Module), 'Only support pytorch module.'

        self.log_dir_root = Path(log_dir)
        self.log_dir_root.mkdir(parents=True, exist_ok=True)

        # init tasks info file, json format {TASK_ID: {PRE_TASK_ID: xxx, SCORE: xxx, LOG_DIR: xxx}}
        self.tasks_info_file = Path(self.log_dir_root, '.tasks')
        with self.tasks_info_file.open(mode='w') as f:
            json_tricks.dump({}, f)

        self.tasks_map: Dict[int, Task] = {}
        self.pending_tasks: List[Task] = []
        self.task_id_candidate = 0

        self.best_score = None
        self.best_task = None

        self.origin_task_id = None

        self._init_origin_task(origin_model, origin_config_list, origin_masks)

    def _init_origin_task(self, origin_model: Module, origin_config_list: Optional[List[Dict]] = None,
                          origin_masks: Optional[Dict[str, Dict[str, Tensor]]] = None):
        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)

        origin_task = Task(task_id, None, deepcopy(origin_config_list), task_log_dir)
        self.tasks_map[task_id] = origin_task
        self.origin_task_id = task_id

        self.task_id_candidate += 1

        self.receive_task_result(task_id, origin_model, origin_masks)

    def receive_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                            score: Optional[float] = None):
        """
        Receive the compressed model, masks and score then save the task result.
        Usually generate new task and put it into `self.pending_tasks` in this function.

        Parameters
        ----------
        task_id
            The id of the task registered in `self.tasks_map`.
        pruned_model
            The pruned model in the last iteration. It might be a sparsify model or a speed-up model.
        masks
            If masks is empty, the pruned model is a compact model after speed up.
            If masks is not None, the pruned model is a sparsify model without speed up.
        score
            The score of the model, higher score means better performance.
        """
        assert task_id in self.tasks_map, 'Task {} does not exist.'.format(task_id)
        task = self.tasks_map[task_id]

        # update the task that has the best score
        if score is not None:
            task.score = score
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_task = task_id

        self._save_task_result(task_id=task_id, pruned_model=pruned_model, masks=masks)

        self.pending_tasks.extend(self._generate_tasks(received_task_id=task_id))

    def _save_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]]):
        """
        Save the task result.

        Parameters
        ----------
        task_id
            The id of the task registered in `self.tasks_map`.
        pruned_model
            The pruned model in the last iteration. It might be a sparsify model or a speed-up model.
        masks
            If masks is empty, the pruned model is a compact model after speed up.
            If masks is not None, the pruned model is a sparsify model without speed up.
        """
        task = self.tasks_map[task_id]

        # save tasks info
        with self.tasks_info_file.open(mode='r') as f:
            tasks_info = json_tricks.load(f)

        with self.tasks_info_file.open(mode='w') as f:
            tasks_info[task_id] = {PRE_TASK_ID: task.pre_task_id, SCORE: task.score, LOG_DIR: task.log_dir,
                                   STATUS: task.status}
            json_tricks.dump(tasks_info, f, indent=4)

        # save config list, pruned model and masks
        with Path(task.log_dir, CONFIG_LIST_NAME).open(mode='w') as f:
            json_tricks.dump(task.config_list, f, indent=4)
        torch.save(pruned_model, Path(task.log_dir, MODEL_NAME))
        torch.save(masks, Path(task.log_dir, MASKS_NAME))

    def load_task_result(self, task_id: int) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        """
        Return the pruned model and masks of the task.
        """
        task = self.tasks_map[task_id]
        model = torch.load(Path(task.log_dir, MODEL_NAME))
        masks = torch.load(Path(task.log_dir, MASKS_NAME))
        return model, masks

    def _generate_tasks(self, received_task_id: int) -> List[Task]:
        """
        Subclass need implement this function to push new tasks into `self.pending_tasks`.
        """
        raise NotImplementedError()

    def next(self) -> Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        """
        Get the next task.

        Returns
        -------
        Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]
            The task id, model, config_list and masks.
        """
        if len(self.pending_tasks) == 0:
            return None, None, None, None
        else:
            task = self.pending_tasks.pop(0)
            model = None
            config_list = deepcopy(task.config_list)
            masks = None
            if task.pre_task_id is not None:
                pre_task = self.tasks_map[task.pre_task_id]
                model = torch.load(Path(pre_task.log_dir, MODEL_NAME))
                masks = torch.load(Path(pre_task.log_dir, MASKS_NAME))
            return task.task_id, model, config_list, masks


class PruningScheduler:
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

    def compress_one_step(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]]) -> Tuple[Module, List[Dict], Optional[float]]:
        # compress model
        self.pruner.reset(model, config_list)
        self.pruner.load_masks(masks)
        model, masks = self.pruner.compress()

        # show the pruning effect
        self.pruner.show_pruned_weights()

        # apply masks to sparsify model
        self.pruner._unwrap_model()
        apply_compression_results(model, masks)

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

    def compress(self):
        iteration = 0
        task_id, model, config_list, masks = self.task_generator.next()

        while task_id is not None:
            pruned_model, masks, score = self.compress_one_step(model, config_list, masks)
            _logger.info('\nIteration %d\ntask id: %d\nscore: %s\nconfig list:\n%s', iteration, task_id, str(score), json_tricks.dumps(config_list, indent=4))

            self.task_generator.receive_task_result(task_id, pruned_model, masks, score)
            task_id, model, config_list, masks = self.task_generator.next()

            iteration += 1
