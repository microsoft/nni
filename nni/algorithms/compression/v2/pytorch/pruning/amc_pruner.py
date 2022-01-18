# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Callable, Optional

import json_tricks
import torch
from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base import Task, TaskResult
from nni.algorithms.compression.v2.pytorch.utils import compute_sparsity, config_list_canonical
from nni.compression.pytorch.utils.counter import count_flops_params

from .iterative_pruner import IterativePruner, PRUNER_DICT
from .tools import TaskGenerator
from .tools.rl_env import DDPG, AMCEnv


class AMCTaskGenerator(TaskGenerator):
    """
    Parameters
    ----------
    total_episode
        The total episode number.
    dummy_input
        Use to inference and count the flops.
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
    ddpg_params
        The ddpg agent parameters.
    target : str
        'flops' or 'params'. Note that the sparsity in other pruners always means the parameters sparse, but in AMC, you can choose flops sparse.
        This parameter is used to explain what the sparsity setting in config_list refers to.
    """

    def __init__(self, total_episode: int, dummy_input: Tensor, origin_model: Module, origin_config_list: List[Dict],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = '.', keep_intermediate_result: bool = False,
                 ddpg_params: Dict = {}, target: str = 'flops'):
        self.total_episode = total_episode
        self.current_episode = 0
        self.dummy_input = dummy_input
        self.ddpg_params = ddpg_params
        self.target = target
        self.config_list_copy = deepcopy(origin_config_list)

        super().__init__(origin_model=origin_model, origin_masks=origin_masks, origin_config_list=origin_config_list,
                         log_dir=log_dir, keep_intermediate_result=keep_intermediate_result)

    def init_pending_tasks(self) -> List[Task]:
        origin_model = torch.load(self._origin_model_path)
        origin_masks = torch.load(self._origin_masks_path)
        with open(self._origin_config_list_path, "r") as f:
            origin_config_list = json_tricks.load(f)

        self.T = []
        self.action = None
        self.observation = None
        self.warmup_episode = self.ddpg_params['warmup'] if 'warmup' in self.ddpg_params.keys() else int(self.total_episode / 4)

        config_list_copy = config_list_canonical(origin_model, origin_config_list)
        total_sparsity = config_list_copy[0]['total_sparsity']
        max_sparsity_per_layer = config_list_copy[0].get('max_sparsity_per_layer', 1.)

        self.env = AMCEnv(origin_model, origin_config_list, self.dummy_input, total_sparsity, max_sparsity_per_layer, self.target)
        self.agent = DDPG(len(self.env.state_feature), 1, self.ddpg_params)
        self.agent.is_training = True
        task_result = TaskResult('origin', origin_model, origin_masks, origin_masks, None)

        return self.generate_tasks(task_result)

    def generate_tasks(self, task_result: TaskResult) -> List[Task]:
        # append experience & update agent policy
        if task_result.task_id != 'origin':
            action, reward, observation, done = self.env.step(self.action, task_result.compact_model)
            self.T.append([reward, self.observation, observation, self.action, done])
            self.observation = observation.copy()

            if done:
                final_reward = task_result.score - 1
                # agent observe and update policy
                for _, s_t, s_t1, a_t, d_t in self.T:
                    self.agent.observe(final_reward, s_t, s_t1, a_t, d_t)
                    if self.current_episode > self.warmup_episode:
                        self.agent.update_policy()

                self.current_episode += 1
                self.T = []
                self.action = None
                self.observation = None

            # update current2origin_sparsity in log file
            origin_model = torch.load(self._origin_model_path)
            compact_model = task_result.compact_model
            compact_model_masks = task_result.compact_model_masks
            current2origin_sparsity, _, _ = compute_sparsity(origin_model, compact_model, compact_model_masks, self.temp_config_list)
            self._tasks[task_result.task_id].state['current2origin_sparsity'] = current2origin_sparsity
            current2origin_sparsity, _, _ = compute_sparsity(origin_model, compact_model, compact_model_masks, self.config_list_copy)
            self._tasks[task_result.task_id].state['current_total_sparsity'] = current2origin_sparsity
            flops, params, _ = count_flops_params(compact_model, self.dummy_input, verbose=False)
            self._tasks[task_result.task_id].state['current_flops'] = '{:.2f} M'.format(flops / 1e6)
            self._tasks[task_result.task_id].state['current_params'] = '{:.2f} M'.format(params / 1e6)

        # generate new action
        if self.current_episode < self.total_episode:
            if self.observation is None:
                self.observation = self.env.reset().copy()
                self.temp_config_list = []
                compact_model = torch.load(self._origin_model_path)
                compact_model_masks = torch.load(self._origin_masks_path)
            else:
                compact_model = task_result.compact_model
                compact_model_masks = task_result.compact_model_masks
            if self.current_episode <= self.warmup_episode:
                action = self.agent.random_action()
            else:
                action = self.agent.select_action(self.observation, episode=self.current_episode)
            action = action.tolist()[0]

            self.action = self.env.correct_action(action, compact_model)
            sub_config_list = [{'op_names': [self.env.current_op_name], 'total_sparsity': self.action}]
            self.temp_config_list.extend(sub_config_list)

            task_id = self._task_id_candidate
            if self.env.is_first_layer() or self.env.is_final_layer():
                task_config_list = self.temp_config_list
            else:
                task_config_list = sub_config_list

            config_list_path = Path(self._intermediate_result_dir, '{}_config_list.json'.format(task_id))
            with Path(config_list_path).open('w') as f:
                json_tricks.dump(task_config_list, f, indent=4)

            model_path = Path(self._intermediate_result_dir, '{}_compact_model.pth'.format(task_result.task_id))
            masks_path = Path(self._intermediate_result_dir, '{}_compact_model_masks.pth'.format(task_result.task_id))
            torch.save(compact_model, model_path)
            torch.save(compact_model_masks, masks_path)

            task = Task(task_id, model_path, masks_path, config_list_path)
            if not self.env.is_final_layer():
                task.finetune = False
                task.evaluate = False

            self._tasks[task_id] = task
            self._task_id_candidate += 1
            return [task]
        else:
            return []


class AMCPruner(IterativePruner):
    """
    A pytorch implementation of AMC: AutoML for Model Compression and Acceleration on Mobile Devices.
    (https://arxiv.org/pdf/1802.03494.pdf)
    Suggust config all `total_sparsity` in `config_list` a same value.
    AMC pruner will treat the first sparsity in `config_list` as the global sparsity.

    Parameters
    ----------
    total_episode : int
        The total episode number.
    model : Module
        The model to be pruned.
    config_list : List[Dict]
        Supported keys :
            - total_sparsity : This is to specify the total sparsity for all layers in this config, each layer may have different sparsity.
            - max_sparsity_per_layer : Always used with total_sparsity. Limit the max sparsity of each layer.
            - op_types : Operation type to be pruned.
            - op_names : Operation name to be pruned.
            - op_partial_names: Operation partial names to be pruned, will be autocompleted by NNI.
            - exclude  : Set True then the layers setting by op_types and op_names will be excluded from pruning.
    dummy_input : torch.Tensor
        `dummy_input` is required for speed-up and tracing the model in RL environment.
    evaluator : Callable[[Module], float]
        Evaluate the pruned model and give a score.
    pruning_algorithm : str
        Supported pruning algorithm ['l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    ddpg_params : Dict
        Configuration dict to configure the DDPG agent, any key unset will be set to default implicitly.
        - hidden1: hidden num of first fully connect layer. Default: 300
        - hidden2: hidden num of second fully connect layer. Default: 300
        - lr_c: learning rate for critic. Default: 1e-3
        - lr_a: learning rate for actor. Default: 1e-4
        - warmup: number of episodes without training but only filling the replay memory. During warmup episodes, random actions ares used for pruning. Default: 100
        - discount: next Q value discount for deep Q value target. Default: 0.99
        - bsize: minibatch size for training DDPG agent. Default: 64
        - rmsize: memory size for each layer. Default: 100
        - window_length: replay buffer window length. Default: 1
        - tau: moving average for target network being used by soft_update. Default: 0.99
        - init_delta: initial variance of truncated normal distribution. Default: 0.5
        - delta_decay: delta decay during exploration. Default: 0.99
        # parameters for training ddpg agent
        - max_episode_length: maximum episode length. Default: 1e9
        - epsilon: linear decay of exploration policy. Default: 50000

    pruning_params : Dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    target : str
        'flops' or 'params'. Note that the sparsity in other pruners always means the parameters sparse, but in AMC, you can choose flops sparse.
        This parameter is used to explain what the sparsity setting in config_list refers to.
    """

    def __init__(self, total_episode: int, model: Module, config_list: List[Dict], dummy_input: Tensor,
                 evaluator: Callable[[Module], float], pruning_algorithm: str = 'l1', log_dir: str = '.',
                 keep_intermediate_result: bool = False, finetuner: Optional[Callable[[Module], None]] = None,
                 ddpg_params: dict = {}, pruning_params: dict = {}, target: str = 'flops'):
        assert pruning_algorithm in ['l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo'], \
            "Only support pruning_algorithm in ['l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo']"
        task_generator = AMCTaskGenerator(total_episode=total_episode,
                                          dummy_input=dummy_input,
                                          origin_model=model,
                                          origin_config_list=config_list,
                                          log_dir=log_dir,
                                          keep_intermediate_result=keep_intermediate_result,
                                          ddpg_params=ddpg_params,
                                          target=target)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=True, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)
