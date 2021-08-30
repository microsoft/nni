# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json_tricks

import numpy as np
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.utils.pruning import config_list_canonical, compute_sparsity, get_model_weights_numel
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

    def _init_pending_tasks(self) -> List[Task]:
        model, _, masks = self._load_origin_data()
        return self._generate_tasks(None, model, masks)

    def _generate_tasks(self, received_task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]]) -> List[Task]:
        origin_model, origin_config_list, _ = self._load_origin_data()

        real_sparsity, mo_sparsity, _ = compute_sparsity(origin_model, pruned_model, masks, origin_config_list)
        _logger.info('Task %s total real sparsity compared with original model is:\n%s', str(received_task_id), json_tricks.dumps(real_sparsity, indent=4))

        # if reach the total_iteration, no more task will be generated
        if self.current_iteration >= self.total_iteration:
            return []
        config_list = self._generate_config_list(self.target_sparsity, self.current_iteration, mo_sparsity)

        task_id = self._task_id_candidate
        if self._save_result:
            task_log_dir = Path(self.log_dir_root, str(task_id))
            task_log_dir.mkdir(parents=True, exist_ok=True)
        else:
            task_log_dir = None

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


class SimulatedAnnealingTaskGenerator(TaskGenerator):
    def __init__(self, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.',
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, save_result: bool = False):
        assert all(config.get('total_sparsity') is not None for config in origin_config_list), 'Only support total_sparsity in simulated annealing.'

        self.start_temperature = start_temperature
        self.current_temperature = start_temperature
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate
        self.perturbation_magnitude = perturbation_magnitude

        self.weights_numel, self.masked_rate = get_model_weights_numel(origin_model, origin_config_list, origin_masks)
        self.target_sparsity_list = config_list_canonical(origin_model, origin_config_list)
        self._adjust_target_sparsity()

        self._temp_config_list = None
        self._current_sparsity_list = None
        self._current_score = None

        super().__init__(origin_model, origin_config_list=origin_config_list, origin_masks=origin_masks,
                         log_dir=log_dir, save_result=save_result)

    def _init_pending_tasks(self) -> List[Task]:
        model, _, masks = self._load_origin_data()
        return self._generate_tasks(None, model, masks)

    def _adjust_target_sparsity(self):
        """
        If origin_masks is not empty, then re-scale the target sparsity.
        """
        if len(self.masked_rate) > 0:
            for config in self.target_sparsity_list:
                sparsity, op_names = config['total_sparsity'], config['op_names']
                remaining_weight_numel = 0
                pruned_weight_numel = 0
                for name in op_names:
                    remaining_weight_numel += self.weights_numel[name]
                    if name in self.masked_rate:
                        pruned_weight_numel += 1 / (1 / self.masked_rate[name] - 1) * self.weights_numel[name]
                config['total_sparsity'] = max(0, sparsity - pruned_weight_numel / (pruned_weight_numel + remaining_weight_numel))

    def _recover_real_sparsity(self, config_list: List[Dict]) -> List[Dict]:
        """
        Recover the generated config_list if origin_masks is not empty.
        """
        for config in config_list:
            assert len(config['op_names']) == 1
            op_name = config['op_names'][0]
            if op_name in self.masked_rate:
                config['total_sparsity'] = self.masked_rate[op_name] + config['total_sparsity'] * (1 - self.masked_rate[op_name])
        return config_list

    def _init_temp_config_list(self):
        self._temp_config_list = []
        self._temp_sparsity_list = []
        for config in self.target_sparsity_list:
            sparsity_config, sparsity = self._init_config_sparsity(config)
            self._temp_config_list.extend(sparsity_config)
            self._temp_sparsity_list.append(sparsity)

    def _init_config_sparsity(self, config: Dict) -> Tuple[List[Dict], List]:
        assert 'total_sparsity' in config, 'Sparsity must be set in config: {}'.format(config)
        target_sparsity = config['total_sparsity']
        op_names = config['op_names']

        if target_sparsity == 0:
            return [], []

        while True:
            random_sparsity = sorted(np.random.uniform(0, 1, len(op_names)))
            rescaled_sparsity = self._rescale_sparsity(random_sparsity, target_sparsity, op_names)
            if rescaled_sparsity is not None and rescaled_sparsity[0] >= 0 and rescaled_sparsity[-1] < 1:
                break

        return self._sparsity_to_config_list(rescaled_sparsity, config), rescaled_sparsity

    def _rescale_sparsity(self, random_sparsity: List, target_sparsity: float, op_names: List) -> List:
        assert len(random_sparsity) == len(op_names)

        num_weights = sorted([self.weights_numel[op_name] for op_name in op_names])
        sparsity = sorted(random_sparsity)

        total_weights = 0
        total_weights_pruned = 0

        # calculate the scale
        for idx, num_weight in enumerate(num_weights):
            total_weights += num_weight
            total_weights_pruned += int(num_weight * sparsity[idx])
        if total_weights_pruned == 0:
            return None

        scale = target_sparsity / (total_weights_pruned / total_weights)

        # rescale the sparsity
        sparsity = np.asarray(sparsity) * scale
        return sparsity

    def _sparsity_to_config_list(self, sparsity: List, config: Dict) -> List[Dict]:
        sparsity = sorted(sparsity)
        op_names = [k for k, _ in sorted(self.weights_numel.items(), key=lambda item: item[1]) if k in config['op_names']]
        return [{'total_sparsity': sparsity, 'op_names': [op_name]} for sparsity, op_name in zip(sparsity, op_names)]

    def _update_with_perturbations(self):
        self._temp_config_list = []
        self._temp_sparsity_list = []
        # decrease magnitude with current temperature
        magnitude = self.current_temperature / self.start_temperature * self.perturbation_magnitude
        for config, current_sparsity in zip(self.target_sparsity_list, self._current_sparsity_list):
            if len(current_sparsity) == 0:
                self._temp_sparsity_list.append([])
                continue
            while True:
                perturbation = np.random.uniform(-magnitude, magnitude, len(current_sparsity))
                temp_sparsity = np.clip(0, current_sparsity + perturbation, None)
                temp_sparsity = self._rescale_sparsity(temp_sparsity, config['total_sparsity'], config['op_names'])
                if temp_sparsity is not None and temp_sparsity[0] >= 0 and temp_sparsity[-1] < 1:
                    self._temp_config_list.extend(self._sparsity_to_config_list(temp_sparsity, config))
                    self._temp_sparsity_list.append(temp_sparsity)
                    break

    def receive_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                            score: Optional[float] = None):
        assert score is not None, 'score can not be None in simulated annealing.'
        return super().receive_task_result(task_id, pruned_model, masks, score=score)

    def _generate_tasks(self, received_task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]]) -> List[Task]:
        if self._temp_config_list is None:
            self._init_temp_config_list()
        else:
            score = self._tasks[received_task_id].score
            if self._current_sparsity_list is None:
                self._current_sparsity_list = deepcopy(self._temp_sparsity_list)
                self._current_score = score
            else:
                delta_E = np.abs(score - self._current_score)
                probability = np.exp(-1 * delta_E / self.current_temperature)
                if self._current_score < score or np.random.uniform(0, 1) < probability:
                    self._current_score = score
                    self._current_sparsity_list = deepcopy(self._temp_sparsity_list)
                    self.current_temperature *= self.cool_down_rate
            if self.current_temperature < self.stop_temperature:
                return []
            self._update_with_perturbations()

        task_id = self._task_id_candidate
        if self._save_result:
            task_log_dir = Path(self.log_dir_root, str(task_id))
            task_log_dir.mkdir(parents=True, exist_ok=True)
        else:
            task_log_dir = None

        config_list = self._recover_real_sparsity(deepcopy(self._temp_config_list))

        task = Task(self._task_id_candidate, Path(self._log_dir_root, 'origin', 'origin_model.pth'), config_list,
                    Path(self._log_dir_root, 'origin', 'origin_masks.pth'), log_dir=task_log_dir)
        self._tasks[task_id] = task

        self._task_id_candidate += 1

        return [task]
