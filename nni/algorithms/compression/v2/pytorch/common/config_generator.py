# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base.scheduler import Task, TaskGenerator
from nni.algorithms.compression.v2.pytorch.utils.pruning import unfold_config_list, dedupe_config_list, compute_sparsity


class ConsistentTaskGenerator(TaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.'):
        self.total_iteration = total_iteration
        super().__init__(origin_model, origin_config_list=origin_config_list, origin_masks=origin_masks,
                         log_dir=log_dir)

    def _init_origin_task(self, origin_model: Module, origin_config_list: Optional[List[Dict]] = None,
                          origin_masks: Optional[Dict[str, Dict[str, Tensor]]] = None):
        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)

        target_sparsity = dedupe_config_list(unfold_config_list(origin_model, origin_config_list))
        assert all('sparsity' in config for config in target_sparsity), 'Sparsity is needed in AGP, please specify sparsity for each config.'
        pre_real_sparsity, _, _ = compute_sparsity(origin_model, origin_model, origin_masks, origin_config_list)
        status = {
            'current_iteration': -1,
            'target_sparsity': target_sparsity,
            'pre_real_sparsity': pre_real_sparsity
        }

        origin_task = Task(task_id, None, deepcopy(origin_config_list), task_log_dir, status=status)
        self.tasks_map[task_id] = origin_task
        self.origin_task_id = task_id

        self.task_id_candidate += 1

        self.receive_task_result(task_id, origin_model, origin_masks)

    def _generate_tasks(self, pre_task_id: int) -> List[Task]:
        pre_task = self.tasks_map[pre_task_id]
        pruned_model, masks = self.load_task_result(pre_task_id)

        origin_model, _ = self.load_task_result(self.origin_task_id)
        origin_config_list = self.tasks_map[self.origin_task_id].config_list

        pre_real_sparsity, mo_sparsity, _ = compute_sparsity(origin_model, pruned_model, masks, origin_config_list)

        target_sparsity = self.tasks_map[self.origin_task_id].status['target_sparsity']
        current_iteration = pre_task.status['current_iteration'] + 1
        # if reach the total_iteration, no more task will be generated
        if current_iteration > self.total_iteration:
            return []
        config_list = self._generate_config_list(target_sparsity, current_iteration, mo_sparsity)

        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)
        status = {
            'current_iteration': current_iteration,
            'target_sparsity': target_sparsity,
            'pre_real_sparsity': pre_real_sparsity
        }
        task = Task(task_id, pre_task_id, config_list, task_log_dir, status=status)
        self.tasks_map[task_id] = task

        self.task_id_candidate += 1

        return [task]

    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        raise NotImplementedError()


class AGPTaskGenerator(ConsistentTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = (1 - (1 - iteration / self.total_iteration) ** 3) * target['sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['sparsity']) / (1 - mo['sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['sparsity'] = sparsity
        return config_list


class LinearTaskGenerator(ConsistentTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = iteration / self.total_iteration * target['sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['sparsity']) / (1 - mo['sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['sparsity'] = sparsity
        return config_list
