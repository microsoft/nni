from copy import deepcopy
import logging
from typing import Callable, List, Dict, Tuple

import numpy as np
from torch.nn import Module

from .sparsity_generator import SparsityScheduler, SparsityAllocator, SparsityGenerator
from .naive_sparsity_genrator import NaiveSparsityScheduler
from utils import unfold_config_list, dedupe_config_list, get_model_weight_numel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulatedAnnealingAllocator(SparsityAllocator):
    def __init__(self, model: Module, config_list: List[Dict], evaluator: Callable[[Module], float],
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35):
        self._evaluator = evaluator

        self._start_temperature = start_temperature
        self._current_temperature = start_temperature
        self._stop_temperature = stop_temperature
        self._cool_down_rate = cool_down_rate
        self._perturbation_magnitude = perturbation_magnitude

        self._unfolded_config_list = dedupe_config_list(unfold_config_list(model, config_list))
        self._update_model_weight_numel(model)

        self._temp_config_list = None
        self._temp_sparsities_list = None

        self._best_config_list = None
        self._best_score = None

        self._current_score = None
        self._current_sparsities_list = None

    def _update_model_weight_numel(self, model: Module):
        self._model_weight_numel = get_model_weight_numel(model, self._unfolded_config_list)

    def _init_sparsities_config_list(self):
        self._temp_config_list = []
        self._temp_sparsities_list = []
        for config in self._unfolded_config_list:
            sparsities_config, sparsities = self._init_config_sparsity(config)
            self._temp_config_list.extend(sparsities_config)
            self._temp_sparsities_list.append(sparsities)

    def _init_config_sparsity(self, config: Dict) -> Tuple[List[Dict], List]:
        target_sparsity = config['sparsity']
        op_names = config['op_names']

        while True:
            sparsities = sorted(np.random.uniform(0, 1, len(op_names)))
            sparsities = self._rescale_sparsities(sparsities, target_sparsity, op_names)
            if sparsities is not None and sparsities[0] >= 0 and sparsities[-1] < 1:
                break

        return self._sparsities_to_config_list(sparsities, config), sparsities

    def _rescale_sparsities(self, sparsities: List, target_sparsity: float, op_names: List) -> List:
        assert len(sparsities) == len(op_names)

        num_weights = sorted([self._model_weight_numel[op_name] for op_name in op_names])
        sparsities = sorted(sparsities)

        total_weights = 0
        total_weights_pruned = 0

        # calculate the scale
        for idx, num_weight in enumerate(num_weights):
            total_weights += num_weight
            total_weights_pruned += int(num_weight * sparsities[idx])
        if total_weights_pruned == 0:
            return None

        scale = target_sparsity / (total_weights_pruned / total_weights)

        # rescale the sparsities
        sparsities = np.asarray(sparsities) * scale
        return sparsities

    def _sparsities_to_config_list(self, sparsities: List, config: Dict) -> List[Dict]:
        sparsities = sorted(sparsities)
        op_names = [k for k, _ in sorted(self._model_weight_numel.items(), key=lambda item: item[1]) if k in config['op_names']]
        return [{'sparsity': sparsity, 'op_names': [op_name]} for sparsity, op_name in zip(sparsities, op_names)]

    def _update_with_perturbations(self):
        '''
        Generate perturbation to the current sparsities distribution.
        '''
        self._temp_config_list = []
        self._temp_sparsities_list = []
        # decrease magnitude with current temperature
        magnitude = self._current_temperature / self._start_temperature * self._perturbation_magnitude
        for config, sparsities in zip(self._unfolded_config_list, self._current_sparsities_list):
            while True:
                perturbation = np.random.uniform(-magnitude, magnitude, len(sparsities))
                temp_sparsities = np.clip(0.01, sparsities + perturbation, None)
                temp_sparsities = self._rescale_sparsities(temp_sparsities, config['sparsity'], config['op_names'])
                if temp_sparsities is not None and temp_sparsities[0] >= 0 and temp_sparsities[-1] < 1:
                    self._temp_config_list.extend(self._sparsities_to_config_list(temp_sparsities, config))
                    self._temp_sparsities_list.append(temp_sparsities)
                    break

    def _trans_sparsities_config_list(self) -> List[Dict]:
        config_list = []
        for sparsities_config in self._temp_sparsities_list:
            config_list.extend(sparsities_config)
        return config_list

    def get_allocated_config_list(self, model: Module, schedule_config_list: List[Dict]) -> List[Dict]:
        if self._temp_sparsities_list is None:
            self._init_sparsities_config_list()
        else:
            score = self._evaluator(model)
            if self._best_score is None:
                self._best_score = self._current_score = score
                self._best_config_list = deepcopy(self._temp_config_list)
                self._current_sparsities_list = deepcopy(self._temp_sparsities_list)
            else:
                delta_E = np.abs(score - self._current_score)
                probability = np.exp(-1 * delta_E / self._current_temperature)
                if self._best_score < score:
                    self._best_score = score
                    self._best_config_list = deepcopy(self._temp_config_list)
                if self._current_score < score or np.random.uniform(0, 1) < probability:
                    self._current_score = score
                    self._current_sparsities_list = deepcopy(self._temp_sparsities_list)
                    self._current_temperature *= self._cool_down_rate
            if self._current_temperature < self._stop_temperature:
                return None
            self._update_with_perturbations()

        return deepcopy(self._temp_config_list)


class SimulatedAnnealingSparsityGenerator(SparsityGenerator):
    def __init__(self, model: Module, config_list: List[Dict], evaluator: Callable[[Module], float],
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35):
        self._sparsity_scheduler = NaiveSparsityScheduler(config_list, None)
        self._sparsity_allocator = SimulatedAnnealingAllocator(model, config_list, evaluator, start_temperature,
                                                               stop_temperature, cool_down_rate, perturbation_magnitude)

    @property
    def sparsity_scheduler(self) -> SparsityScheduler:
        return self._sparsity_scheduler

    @property
    def sparsity_allocator(self) -> SparsityAllocator:
        return self._sparsity_allocator

    @property
    def best_config_list(self) -> List[Dict]:
        return self._sparsity_allocator._best_config_list
