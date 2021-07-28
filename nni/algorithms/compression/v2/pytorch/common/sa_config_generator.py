# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base.scheduler import Task, TaskGenerator
from nni.algorithms.compression.v2.pytorch.utils.pruning import unfold_config_list, dedupe_config_list, get_model_weights_numel


class SimulatedAnnealingTaskGenerator(TaskGenerator):
    def __init__(self, origin_model: Module, origin_config_list: List[Dict] = [],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.',
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35):
        self.start_temperature = start_temperature
        self.current_temperature = start_temperature
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate
        self.perturbation_magnitude = perturbation_magnitude

        self.weights_numel, self.masked_rate = get_model_weights_numel(origin_model, origin_model, origin_masks)
        self.target_sparsity_list = dedupe_config_list(unfold_config_list(origin_model, origin_config_list))
        self._adjust_target_sparsity()

        self._temp_config_list = None
        self._current_sparsity_list = None
        self._current_score = None

        super().__init__(origin_model, origin_config_list=origin_config_list, origin_masks=origin_masks,
                         log_dir=log_dir)

    def _adjust_target_sparsity(self):
        if len(self.masked_rate) > 0:
            for config in self.target_sparsity_list:
                sparsity, op_names = config['sparsity'], config['op_names']
                remaining_weight_numel = 0
                pruned_weight_numel = 0
                for name in op_names:
                    remaining_weight_numel += self.weights_numel[name]
                    if name in self.masked_rate:
                        pruned_weight_numel += 1 / (1 / self.masked_rate[name] - 1) * self.weights_numel[name]
                config['sparsity'] = max(0, sparsity - pruned_weight_numel / (pruned_weight_numel + remaining_weight_numel))

    def _init_temp_config_list(self):
        self._temp_config_list = []
        self._temp_sparsity_list = []
        for config in self.target_sparsity_list:
            sparsity_config, sparsity = self._init_config_sparsity(config)
            self._temp_config_list.extend(sparsity_config)
            self._temp_sparsity_list.append(sparsity)

    def _init_config_sparsity(self, config: Dict) -> Tuple[List[Dict], List]:
        assert 'sparsity' in config, 'Sparsity must be set in config: {}'.format(config)
        target_sparsity = config['sparsity']
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
        return [{'sparsity': sparsity, 'op_names': [op_name]} for sparsity, op_name in zip(sparsity, op_names)]

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
                temp_sparsity = np.clip(0.01, current_sparsity + perturbation, None)
                temp_sparsity = self._rescale_sparsity(temp_sparsity, config['sparsity'], config['op_names'])
                if temp_sparsity is not None and temp_sparsity[0] >= 0 and temp_sparsity[-1] < 1:
                    self._temp_config_list.extend(self._sparsity_to_config_list(temp_sparsity, config))
                    self._temp_sparsity_list.append(temp_sparsity)
                    break

    def receive_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]],
                            score: Optional[float]):
        return super().receive_task_result(task_id, pruned_model, masks, score=score)

    def _generate_tasks(self, received_task_id: int) -> List[Task]:
        if self._temp_config_list is None:
            self._init_temp_config_list()
        else:
            score = self.tasks_map[received_task_id].score
            if self._current_sparsity_list is None:
                self._current_sparsity_list = deepcopy(self._temp_sparsity_list)
                self._current_score = score
            else:
                delta_E = np.abs(score - self._current_score)
                probability = np.exp(-1 * delta_E / self._current_temperature)
                if self._current_score < score or np.random.uniform(0, 1) < probability:
                    self._current_score = score
                    self._current_sparsity_list = deepcopy(self._temp_sparsity_list)
                    self.current_temperature *= self.cool_down_rate
            if self.current_temperature < self.stop_temperature:
                return []
            self._update_with_perturbations()

        task_id = self.task_id_candidate
        task_log_dir = Path(self.log_dir_root, str(task_id))
        task_log_dir.mkdir(parents=True, exist_ok=True)

        task = Task(self.task_id_candidate, self.origin_task_id, deepcopy(self._temp_config_list))

        self.task_id_candidate += 1

        return [task]
