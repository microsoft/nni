# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List
import json_tricks

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils.pruning import unfold_config_list, dedupe_config_list, compute_sparsity
from .base import Task, TaskGenerator

_logger = logging.getLogger(__name__)


class ConsistentTaskGenerator(TaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.'):
        self.current_iteration = 0
        self.target_sparsity = dedupe_config_list(unfold_config_list(origin_model, origin_config_list))
        self.total_iteration = total_iteration

        super().__init__(origin_model, origin_config_list=origin_config_list, origin_masks=origin_masks,
                         log_dir=log_dir)

    def _generate_tasks(self, received_task_id: int) -> List[Task]:
        pruned_model, masks = self.load_task_result(received_task_id)

        origin_model, _ = self.load_task_result(self.origin_task_id)
        origin_config_list = self.tasks_map[self.origin_task_id].config_list

        real_sparsity, mo_sparsity, _ = compute_sparsity(origin_model, pruned_model, masks, origin_config_list)
        _logger.info('Task %s total real sparsity compared with original model is:\n%s', str(received_task_id), json_tricks.dumps(real_sparsity, indent=4))

        # if reach the total_iteration, no more task will be generated
        if self.current_iteration >= self.total_iteration:
            return []
        config_list = self._generate_config_list(self.target_sparsity, self.current_iteration, mo_sparsity)

        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)

        task = Task(task_id, received_task_id, config_list, task_log_dir)
        self.tasks_map[task_id] = task

        self.task_id_candidate += 1
        self.current_iteration += 1

        return [task]

    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        raise NotImplementedError()


class AGPTaskGenerator(ConsistentTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = (1 - (1 - iteration / self.total_iteration) ** 3) * target['_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['_sparsity']) / (1 - mo['_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['_sparsity'] = sparsity
        return config_list


class LinearTaskGenerator(ConsistentTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = iteration / self.total_iteration * target['_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['_sparsity']) / (1 - mo['_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['_sparsity'] = sparsity
        return config_list
