# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List
import json_tricks

from torch import Tensor
import torch
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Task, TaskResult
from nni.algorithms.compression.v2.pytorch.utils.pruning import config_list_canonical, compute_sparsity
from .base import TaskGenerator

_logger = logging.getLogger(__name__)


class FunctionBasedTaskGenerator(TaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', keep_intermidiate_result: bool = False):
        self.current_iteration = 0
        self.target_sparsity = config_list_canonical(origin_model, origin_config_list)
        self.total_iteration = total_iteration

        super().__init__(origin_model, origin_config_list=self.target_sparsity, origin_masks=origin_masks,
                         log_dir=log_dir, keep_intermidiate_result=keep_intermidiate_result)

    def init_pending_tasks(self) -> List[Task]:
        origin_model = torch.load(self._origin_model_path)
        origin_masks = torch.load(self._origin_masks_path)

        task_result = TaskResult('origin', origin_model, origin_masks, origin_masks, None)

        return self.generate_tasks(task_result)

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        compact_model = task_result.compact_model
        compact_model_masks = task_result.compact_model_masks

        # save intermidiate result
        model_path = Path(self._intermidiate_result_dir, '{}_compact_model.pth'.format(task_result.task_id))
        masks_path = Path(self._intermidiate_result_dir, '{}_compact_model_masks.pth'.format(task_result.task_id))
        torch.save(compact_model, model_path)
        torch.save(compact_model_masks, masks_path)

        # get current2origin_sparsity and compact2origin_sparsity
        origin_model = torch.load(self._origin_model_path)
        current2origin_sparsity, compact2origin_sparsity, _ = compute_sparsity(origin_model, compact_model, compact_model_masks, self.target_sparsity)
        _logger.info('\nTask %s total real sparsity compared with original model is:\n%s', str(task_result.task_id), json_tricks.dumps(current2origin_sparsity, indent=4))
        if task_result.task_id != 'origin':
            self._tasks[task_result.task_id].state['current2origin_sparsity'] = current2origin_sparsity

        # if reach the total_iteration, no more task will be generated
        if self.current_iteration >= self.total_iteration:
            return []

        task_id = self._task_id_candidate
        new_config_list = self.generate_config_list(self.target_sparsity, self.current_iteration, compact2origin_sparsity)
        config_list_path = Path(self._intermidiate_result_dir, '{}_config_list.json'.format(task_id))

        with Path(config_list_path).open('w') as f:
            json_tricks.dump(new_config_list, f, indent=4)
        task = Task(task_id, model_path, masks_path, config_list_path)

        self._tasks[task_id] = task

        self._task_id_candidate += 1
        self.current_iteration += 1

        return [task]

    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, compact2origin_sparsity: List[Dict]) -> List[Dict]:
        raise NotImplementedError()


class AGPTaskGenerator(FunctionBasedTaskGenerator):
    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = (1 - (1 - iteration / self.total_iteration) ** 3) * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list


class LinearTaskGenerator(FunctionBasedTaskGenerator):
    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = iteration / self.total_iteration * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list
