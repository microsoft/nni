# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from copy import deepcopy
from typing import Dict, List, Callable, Optional

from torch import Tensor
from torch.nn import Module

from .basic_pruner import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    ActivationAPoZRankPruner,
    ActivationMeanRankPruner,
    TaylorFOWeightPruner,
    ADMMPruner
)
from .basic_scheduler import PruningScheduler
from .tools import (
    LinearTaskGenerator,
    AGPTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator,
    AMCTaskGenerator
)

_logger = logging.getLogger(__name__)

__all__ = ['LinearPruner', 'AGPPruner', 'LotteryTicketPruner', 'SimulatedAnnealingPruner', 'AMCPruner']


PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1NormPruner,
    'l2': L2NormPruner,
    'fpgm': FPGMPruner,
    'slim': SlimPruner,
    'apoz': ActivationAPoZRankPruner,
    'mean_activation': ActivationMeanRankPruner,
    'taylorfo': TaylorFOWeightPruner,
    'admm': ADMMPruner
}


class IterativePruner(PruningScheduler):
    def _wrap_model(self):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            This pruner is an iterative pruner and does not directly wrap the model.')

    def _unwrap_model(self):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            This pruner is an iterative pruner and does not directly wrap the model.')

    def export_model(self, *args, **kwargs):
        """
        Deprecated function.
        """
        _logger.warning('Nothing will happen when calling this function.\
            The best result (and intermediate result if keeped) during iteration is under `log_dir` (default: \\.).')


class LinearPruner(IterativePruner):
    """
    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration, usually for neutralizing the accuracy loss brought by the pruning in this iteration.
    speed_up : bool
        If set True, speed up the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for tracing the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: Dict = {}):
        task_generator = LinearTaskGenerator(total_iteration=total_iteration,
                                             origin_model=model,
                                             origin_config_list=config_list,
                                             log_dir=log_dir,
                                             keep_intermediate_result=keep_intermediate_result)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)


class AGPPruner(IterativePruner):
    """
    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration, usually for neutralizing the accuracy loss brought by the pruning in this iteration.
    speed_up : bool
        If set True, speed up the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for tracing the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: Dict = {}):
        task_generator = AGPTaskGenerator(total_iteration=total_iteration,
                                          origin_model=model,
                                          origin_config_list=config_list,
                                          log_dir=log_dir,
                                          keep_intermediate_result=keep_intermediate_result)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)


class LotteryTicketPruner(IterativePruner):
    """
    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration : int
        The total iteration number.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input.
        It will be called at the end of each iteration if reset_weight is False, will be called at the beginning of each iteration otherwise.
    speed_up : bool
        If set True, speed up the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for tracing the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    reset_weight : bool
        If set True, the model weight will reset to the original model weight at the end of each iteration step.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, reset_weight: bool = True,
                 pruning_params: Dict = {}):
        task_generator = LotteryTicketTaskGenerator(total_iteration=total_iteration,
                                                    origin_model=model,
                                                    origin_config_list=config_list,
                                                    log_dir=log_dir,
                                                    keep_intermediate_result=keep_intermediate_result)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=reset_weight)


class SimulatedAnnealingPruner(IterativePruner):
    """
    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user.
    evaluator : Callable[[Module], float]
        Evaluate the pruned model and give a score.
    start_temperature : float
        Start temperature of the simulated annealing process.
    stop_temperature : float
        Stop temperature of the simulated annealing process.
    cool_down_rate : float
        Cool down rate of the temperature.
    perturbation_magnitude : float
        Initial perturbation magnitude to the sparsities. The magnitude decreases with current temperature.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    pruning_params : Dict
        If the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model at the end of each iteration to make the pruned model compact.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for tracing the model in speed up.
    """

    def __init__(self, model: Module, config_list: List[Dict], evaluator: Callable[[Module], float], start_temperature: float = 100,
                 stop_temperature: float = 20, cool_down_rate: float = 0.9, perturbation_magnitude: float = 0.35,
                 pruning_algorithm: str = 'level', pruning_params: Dict = {}, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None):
        task_generator = SimulatedAnnealingTaskGenerator(origin_model=model,
                                                         origin_config_list=config_list,
                                                         start_temperature=start_temperature,
                                                         stop_temperature=stop_temperature,
                                                         cool_down_rate=cool_down_rate,
                                                         perturbation_magnitude=perturbation_magnitude,
                                                         log_dir=log_dir,
                                                         keep_intermediate_result=keep_intermediate_result)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)


class AMCPruner(IterativePruner):
    """
    Parameters
    ----------
    model: Module
            The model to be pruned.
    config_list: list
        Configuration list to configure layer pruning.
        Supported keys:
        - op_types: operation type to be pruned
        - op_names: operation name to be pruned
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
    total_iteration: int
        The total iteration number.
    dummy_input : Optional[torch.Tensor]
        `dummy_input` is required for trace the model in RL environment.
    evaluator: Callable[[Module], float]
        Evaluate the pruned model and give a score.
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model in each iteration.

    env_params: dict
        Configuration dict to configure the environment, any key unset will be set to default implicitly.
        Supported keys:
        - model_type: model type to prune, currently 'mobilenet' and 'mobilenetv2' are supported. Default: mobilenet
        - flops_ratio: preserve flops ratio. Default: 0.5
        - lbound: minimum weight preserve ratio for each layer. Default: 0.2
        - rbound: maximum weight preserve ratio for each layer. Default: 1.0
        # parameters for channel pruning
        - n_calibration_batches: number of batches to extract layer information. Default: 60
        - n_points_per_layer: number of feature points per layer. Default: 10
        - channel_round: round channel to multiple of channel_round. Default: 8
        - batch_size: need to be equivalent to 'DataLoader.batch_size'. Default: 50

    ddpg_params: dict
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

    pruning_params : dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, dummy_input: Tensor,
                 evaluator: Callable[[Module], float], pruning_algorithm: str = 'l1', log_dir: str = '.',
                 keep_intermediate_result: bool = False, finetuner: Optional[Callable[[Module], None]] = None,
                 speed_up: bool = False, env_params: dict = {}, ddpg_params: dict = {}, pruning_params: dict = {}):
        task_generator = AMCTaskGenerator(origin_model=model,
                                          origin_config_list=config_list,
                                          total_iteration=total_iteration,
                                          dummy_input=dummy_input,
                                          log_dir=log_dir,
                                          keep_intermediate_result=keep_intermediate_result,
                                          env_params=env_params,
                                          ddpg_params=ddpg_params)
        pruner = PRUNER_DICT[pruning_algorithm](None, None, **pruning_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)
