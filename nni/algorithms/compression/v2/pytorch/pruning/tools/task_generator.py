# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json_tricks

import numpy as np
from torch import Tensor
import torch
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Task, TaskResult
from nni.algorithms.compression.v2.pytorch.utils import (
    config_list_canonical,
    compute_sparsity,
    get_model_weights_numel
)

from .rl_env.channel_pruning_env import ChannelPruningEnv
from .rl_env.agent import DDPG
from .base import TaskGenerator

_logger = logging.getLogger(__name__)


class FunctionBasedTaskGenerator(TaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', keep_intermediate_result: bool = False):
        """
        Parameters
        ----------
        total_iteration
            The total iteration number.
        origin_model
            The origin unwrapped pytorch model to be pruned.
        origin_config_list
            The origin config list provided by the user. Note that this config_list is directly config the origin model.
            This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
        origin_masks
            The pre masks on the origin model. This mask maybe user-defined or maybe generate by previous pruning.
        log_dir
            The log directory use to saving the task generator log.
        keep_intermediate_result
            If keeping the intermediate result, including intermediate model and masks during each iteration.
        """
        self.total_iteration = total_iteration
        super().__init__(origin_model, origin_config_list=origin_config_list, origin_masks=origin_masks,
                         log_dir=log_dir, keep_intermediate_result=keep_intermediate_result)

    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.current_iteration = 0
        self.target_sparsity = config_list_canonical(model, config_list)
        super().reset(model, config_list=config_list, masks=masks)

    def init_pending_tasks(self) -> List[Task]:
        origin_model = torch.load(self._origin_model_path)
        origin_masks = torch.load(self._origin_masks_path)

        task_result = TaskResult('origin', origin_model, origin_masks, origin_masks, None)

        return self.generate_tasks(task_result)

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        compact_model = task_result.compact_model
        compact_model_masks = task_result.compact_model_masks

        # save intermediate result
        model_path = Path(self._intermediate_result_dir, '{}_compact_model.pth'.format(task_result.task_id))
        masks_path = Path(self._intermediate_result_dir, '{}_compact_model_masks.pth'.format(task_result.task_id))
        torch.save(compact_model, model_path)
        torch.save(compact_model_masks, masks_path)

        # get current2origin_sparsity and compact2origin_sparsity
        origin_model = torch.load(self._origin_model_path)
        current2origin_sparsity, compact2origin_sparsity, _ = compute_sparsity(origin_model, compact_model, compact_model_masks, self.target_sparsity)
        _logger.info('\nTask %s total real sparsity compared with original model is:\n%s', str(task_result.task_id), json_tricks.dumps(current2origin_sparsity, indent=4))
        if task_result.task_id != 'origin':
            self._tasks[task_result.task_id].state['current2origin_sparsity'] = current2origin_sparsity

        # if reach the total_iteration, no more task will be generated
        if self.current_iteration > self.total_iteration:
            return []

        task_id = self._task_id_candidate
        new_config_list = self.generate_config_list(self.target_sparsity, self.current_iteration, compact2origin_sparsity)
        new_config_list = self.allocate_sparsity(new_config_list, compact_model, compact_model_masks)
        config_list_path = Path(self._intermediate_result_dir, '{}_config_list.json'.format(task_id))

        with Path(config_list_path).open('w') as f:
            json_tricks.dump(new_config_list, f, indent=4)
        task = Task(task_id, model_path, masks_path, config_list_path)

        self._tasks[task_id] = task

        self._task_id_candidate += 1
        self.current_iteration += 1

        return [task]

    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, compact2origin_sparsity: List[Dict]) -> List[Dict]:
        raise NotImplementedError()

    def allocate_sparsity(self, new_config_list: List[Dict], model: Module, masks: Dict[str, Dict[str, Tensor]]):
        return new_config_list


class AGPTaskGenerator(FunctionBasedTaskGenerator):
    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, compact2origin_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, compact2origin_sparsity):
            ori_sparsity = (1 - (1 - iteration / self.total_iteration) ** 3) * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list


class LinearTaskGenerator(FunctionBasedTaskGenerator):
    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, compact2origin_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, compact2origin_sparsity):
            ori_sparsity = iteration / self.total_iteration * target['total_sparsity']
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list


class LotteryTicketTaskGenerator(FunctionBasedTaskGenerator):
    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.current_iteration = 1
        self.target_sparsity = config_list_canonical(model, config_list)
        super(FunctionBasedTaskGenerator, self).reset(model, config_list=config_list, masks=masks)

    def generate_config_list(self, target_sparsity: List[Dict], iteration: int, compact2origin_sparsity: List[Dict]) -> List[Dict]:
        config_list = []
        for target, mo in zip(target_sparsity, compact2origin_sparsity):
            # NOTE: The ori_sparsity calculation formula in compression v1 is as follow, it is different from the paper.
            # But the formula in paper will cause numerical problems, so keep the formula in compression v1.
            ori_sparsity = 1 - (1 - target['total_sparsity']) ** (iteration / self.total_iteration)
            # The following is the formula in paper.
            # ori_sparsity = (target['total_sparsity'] * 100) ** (iteration / self.total_iteration) / 100
            sparsity = max(0.0, (ori_sparsity - mo['total_sparsity']) / (1 - mo['total_sparsity']))
            assert 0 <= sparsity <= 1, 'sparsity: {}, ori_sparsity: {}, model_sparsity: {}'.format(sparsity, ori_sparsity, mo['total_sparsity'])
            config_list.append(deepcopy(target))
            config_list[-1]['total_sparsity'] = sparsity
        return config_list


class SimulatedAnnealingTaskGenerator(TaskGenerator):
    def __init__(self, origin_model: Module, origin_config_list: List[Dict], origin_masks: Dict[str, Dict[str, Tensor]] = {},
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, log_dir: str = '.', keep_intermediate_result: bool = False):
        """
        Parameters
        ----------
        origin_model
            The origin unwrapped pytorch model to be pruned.
        origin_config_list
            The origin config list provided by the user. Note that this config_list is directly config the origin model.
            This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
        origin_masks
            The pre masks on the origin model. This mask maybe user-defined or maybe generate by previous pruning.
        start_temperature
            Start temperature of the simulated annealing process.
        stop_temperature
            Stop temperature of the simulated annealing process.
        cool_down_rate
            Cool down rate of the temperature.
        perturbation_magnitude
            Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
        log_dir
            The log directory use to saving the task generator log.
        keep_intermediate_result
            If keeping the intermediate result, including intermediate model and masks during each iteration.
        """
        self.start_temperature = start_temperature
        self.stop_temperature = stop_temperature
        self.cool_down_rate = cool_down_rate
        self.perturbation_magnitude = perturbation_magnitude

        super().__init__(origin_model, origin_masks=origin_masks, origin_config_list=origin_config_list,
                         log_dir=log_dir, keep_intermediate_result=keep_intermediate_result)

    def reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.current_temperature = self.start_temperature

        self.weights_numel, self.masked_rate = get_model_weights_numel(model, config_list, masks)
        self.target_sparsity_list = config_list_canonical(model, config_list)
        self._adjust_target_sparsity()

        self._temp_config_list = None
        self._current_sparsity_list = None
        self._current_score = None

        super().reset(model, config_list=config_list, masks=masks)

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
                    if name in self.masked_rate and self.masked_rate[name] != 0:
                        pruned_weight_numel += 1 / (1 / self.masked_rate[name] - 1) * self.weights_numel[name]
                total_mask_rate = pruned_weight_numel / (pruned_weight_numel + remaining_weight_numel)
                config['total_sparsity'] = max(0, (sparsity - total_mask_rate) / (1 - total_mask_rate))

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

        low_limit = 0
        while True:
            # This is to speed up finding the legal sparsity.
            low_limit = (1 - low_limit) * 0.05 + low_limit
            random_sparsity = sorted(np.random.uniform(low_limit, 1, len(op_names)))
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
        assert len(sparsity) == len(op_names)
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

    def _recover_real_sparsity(self, config_list: List[Dict]) -> List[Dict]:
        """
        If the origin masks is not None, then the sparsity in new generated config_list need to be rescaled.
        """
        for config in config_list:
            assert len(config['op_names']) == 1
            op_name = config['op_names'][0]
            if op_name in self.masked_rate:
                config['total_sparsity'] = self.masked_rate[op_name] + config['total_sparsity'] * (1 - self.masked_rate[op_name])
        return config_list

    def init_pending_tasks(self) -> List[Task]:
        origin_model = torch.load(self._origin_model_path)
        origin_masks = torch.load(self._origin_masks_path)

        self.temp_model_path = Path(self._intermediate_result_dir, 'origin_compact_model.pth')
        self.temp_masks_path = Path(self._intermediate_result_dir, 'origin_compact_model_masks.pth')
        torch.save(origin_model, self.temp_model_path)
        torch.save(origin_masks, self.temp_masks_path)

        task_result = TaskResult('origin', origin_model, origin_masks, origin_masks, None)

        return self.generate_tasks(task_result)

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        # initial/update temp config list
        if self._temp_config_list is None:
            self._init_temp_config_list()
        else:
            score = self._tasks[task_result.task_id].score
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
        new_config_list = self._recover_real_sparsity(deepcopy(self._temp_config_list))
        config_list_path = Path(self._intermediate_result_dir, '{}_config_list.json'.format(task_id))

        with Path(config_list_path).open('w') as f:
            json_tricks.dump(new_config_list, f, indent=4)

        task = Task(task_id, self.temp_model_path, self.temp_masks_path, config_list_path)

        self._tasks[task_id] = task

        self._task_id_candidate += 1

        return [task]


class AMCTaskGenerator(TaskGenerator):
    def __init__(self, origin_model: Module, origin_config_list: List[Dict], total_iteration: int, dummy_input: Tensor,
                 log_dir: str = '.', keep_intermediate_result: bool = False, origin_masks: Dict[str, Dict[str, Tensor]] = {},
                 env_params: Dict = {}, ddpg_params: Dict = {}):
        self.total_iteration = total_iteration
        self.current_iteration = 1
        self.dummy_input = dummy_input
        self.env_params = env_params
        self.ddpg_params = ddpg_params
        self.config_list_copy = deepcopy(origin_config_list)

        super().__init__(origin_model, origin_masks=origin_masks, origin_config_list=origin_config_list,
                         log_dir=log_dir, keep_intermediate_result=keep_intermediate_result)

    def init_pending_tasks(self) -> List[Task]:
        origin_model = torch.load(self._origin_model_path)
        origin_masks = torch.load(self._origin_masks_path)
        with open(self._origin_config_list_path, "r") as f:
            origin_config_list = json_tricks.load(f)

        self.temp_model_path = Path(self._intermediate_result_dir, 'origin_compact_model.pth')
        self.temp_masks_path = Path(self._intermediate_result_dir, 'origin_compact_model_masks.pth')
        torch.save(origin_model, self.temp_model_path)
        torch.save(origin_masks, self.temp_masks_path)

        self.T = []  # trajectory
        self.iteration_reward = 0.
        self.observation = None
        self.actions = []
        self.preserv_idxes = []
        self._temp_config_list = []
        self._current_score = 0
        self.warmup_num = self.ddpg_params['warmup'] if 'warmup' in self.ddpg_params.keys() else 100

        self.env = ChannelPruningEnv(origin_model, origin_config_list, self.dummy_input, self.env_params)
        nb_states = self.env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here
        self.agent = DDPG(nb_states, nb_actions, self.ddpg_params)
        self.agent.is_training = True
        task_result = TaskResult('origin', origin_model, origin_masks, origin_masks, None)

        return self.generate_tasks(task_result)

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        # initial/update temp config list
        if task_result.task_id != 'origin':
            self.env.set_reward(self._tasks[task_result.task_id].score)
            self.env.set_mask(task_result.pruner_generated_masks)

            # use last actions
            info = None
            for action, preserv_idx in zip(self.actions, self.preserv_idxes):
                observation, reward, done, info = self.env.step_after(action, preserv_idx)
                self.T.append([reward, deepcopy(self.observation), deepcopy(observation), action, done])
                # update
                self.observation = deepcopy(observation)
                self.iteration_reward += reward

            # end of episode
            print("Generated config list: {}".format(info['config_list']))
            _logger.info(
                '#%d: iteration_reward: %.4f reward: %.4f, ratio: %.4f\n',
                    self.current_iteration - 1, self.iteration_reward,
                    info['reward'],
                    info['compress_ratio']
            )
            final_reward = self.T[-1][0]
            self._temp_config_list = deepcopy(info['config_list'])

            # agent observe and update policy
            for _, s_t, s_t1, a_t, done in self.T:
                self.agent.observe(final_reward, s_t, s_t1, a_t, done)
                if self.current_iteration > self.warmup_num:
                    self.agent.update_policy()

            # reset
            self.observation = None
            self.iteration_reward = 0.
            self.T = []

            origin_model = torch.load(self._origin_model_path)
            compact_model = task_result.compact_model
            compact_model_masks = task_result.compact_model_masks
            current2origin_sparsity, _, _ = compute_sparsity(origin_model, compact_model, compact_model_masks, self.config_list_copy)
            self._tasks[task_result.task_id].state['current2origin_sparsity'] = current2origin_sparsity
            score = self._tasks[task_result.task_id].score
            # update environment information
            self._current_score = score

        done = False
        self.actions = []
        self.preserv_idxes = []
        self.observation = None
        while not done:
            if self.observation is None:
                self.observation = deepcopy(self.env.reset())
            # agent picks action

            if self.current_iteration <= self.warmup_num:
                action = self.agent.random_action()
            else:
                action = self.agent.select_action(self.observation, episode=self.current_iteration)
            # collect the actions and preserv_idx each layer
            action, preserv_idx = self.env.step_previous(action)
            self.actions.append(action)
            self.preserv_idxes.append(preserv_idx)

            if self.env.cur_ind == len(self.env.prunable_idx):
                self.env.cur_ind = 0
                self.env.visited = [False] * len(self.env.prunable_idx)
                done = True

        if self._task_id_candidate < self.total_iteration:
            task_id = self._task_id_candidate
            config_list_path = Path(self._intermediate_result_dir, '{}_config_list.json'.format(task_id))
            with Path(config_list_path).open('w') as f:
                json_tricks.dump(self._temp_config_list, f, indent=4)

            task = Task(task_id, self.temp_model_path, self.temp_masks_path, config_list_path)
            self._tasks[task_id] = task
            self._task_id_candidate += 1

            # generate task and get task result
            self.current_iteration += 1
            return [task]
        else:
            return []
