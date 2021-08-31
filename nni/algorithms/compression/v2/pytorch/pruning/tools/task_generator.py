# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json_tricks

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils.pruning import config_list_canonical, compute_sparsity
from .base import Task, TaskGenerator

_logger = logging.getLogger(__name__)


class FunctionBasedTaskGenerator(TaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', save_result: bool = True):
        self.current_iteration = 0
        self.target_sparsity = config_list_canonical(origin_model, origin_config_list)
        self.total_iteration = total_iteration

        super().__init__(origin_model, origin_config_list=self.target_sparsity, origin_masks=origin_masks,
                         log_dir=log_dir, save_result=save_result)

    def _save_temp_data(self, temp_model: Module, temp_config_list: List[Dict],
                        temp_masks: Dict[str, Dict[str, Tensor]]):
        self._save_data('temp', temp_model, temp_config_list, temp_masks)

    def _load_temp_data(self) -> Tuple[Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        return self._load_data('temp')

    def _init_pending_tasks(self) -> List[Task]:
        model, _, masks = self._load_origin_data()
        return self._generate_tasks(None, model, masks, None)

    def _generate_tasks(self, received_task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                        origin_masks: Dict[str, Dict[str, Tensor]]) -> List[Task]:
        origin_model, origin_config_list, _ = self._load_origin_data()

        real_sparsity, mo_sparsity, _ = compute_sparsity(origin_model, pruned_model, masks, origin_config_list)
        _logger.info('Task %s total real sparsity compared with original model is:\n%s', str(received_task_id), json_tricks.dumps(real_sparsity, indent=4))

        # if reach the total_iteration, no more task will be generated
        if self.current_iteration >= self.total_iteration:
            return []
        config_list = self._generate_config_list(self.target_sparsity, self.current_iteration, mo_sparsity)

        task_id = self._task_id_candidate
        task_log_dir = Path(self._log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)

        self._save_temp_data(pruned_model, config_list, masks)

        task = Task(task_id, Path(self._log_dir_root, 'temp', 'temp_model.pth'), config_list,
                    Path(self._log_dir_root, 'temp', 'temp_masks.pth'), log_dir=task_log_dir)
        self._tasks[task_id] = task

        self._task_id_candidate += 1
        self.current_iteration += 1

        return [task]

    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        raise NotImplementedError()


class AGPTaskGenerator(FunctionBasedTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = (1 - (1 - iteration / self.total_iteration) ** 3) * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list


class LinearTaskGenerator(FunctionBasedTaskGenerator):
    def _generate_config_list(self, target_sparsity: List[Dict], iteration: int, model_based_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, model_based_sparsity):
            ori_sparsity = iteration / self.total_iteration * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list
