# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
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
from .tools.task_generator import (
    LinearTaskGenerator,
    AGPTaskGenerator,
    LotteryTicketTaskGenerator,
    SimulatedAnnealingTaskGenerator
)

_logger = logging.getLogger(__name__)

__all__ = ['LinearPruner', 'AGPPruner', 'LotteryTicketPruner', 'SimulatedAnnealingPruner']


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
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
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
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model in each iteration.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: dict = {}):
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
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
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
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model in each iteration.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    pruning_params : dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, pruning_params: dict = {}):
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
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
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
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model in each iteration.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
    evaluator : Optional[Callable[[Module], float]]
        Evaluate the pruned model and give a score.
        If evaluator is None, the best result refers to the latest result.
    reset_weight : bool
        If set True, the model weight will reset to the original model weight at the end of each iteration step.
    pruning_params : dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str,
                 total_iteration: int, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 evaluator: Optional[Callable[[Module], float]] = None, reset_weight: bool = True,
                 pruning_params: dict = {}):
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
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
    pruning_algorithm : str
        Supported pruning algorithm ['level', 'l1', 'l2', 'fpgm', 'slim', 'apoz', 'mean_activation', 'taylorfo', 'admm'].
        This iterative pruner will use the chosen corresponding pruner to prune the model in each iteration.
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
    log_dir : str
        The log directory use to saving the result, you can find the best result under this folder.
    keep_intermediate_result : bool
        If keeping the intermediate result, including intermediate model and masks during each iteration.
    finetuner : Optional[Callable[[Module], None]]
        The finetuner handled all finetune logic, use a pytorch module as input, will be called in each iteration.
    speed_up : bool
        If set True, speed up the model in each iteration.
    dummy_input : Optional[torch.Tensor]
        If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
    pruning_params : dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], pruning_algorithm: str, evaluator: Callable[[Module], float],
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 pruning_params: dict = {}):
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
