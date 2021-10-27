# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Dict, List, Callable, Optional

from torch import Tensor
from torch.nn import Module

from .basic_pruner import ADMMPruner
from .iterative_pruner import IterativePruner, SimulatedAnnealingPruner
from .tools import LotteryTicketTaskGenerator


class AutoCompressTaskGenerator(LotteryTicketTaskGenerator):
    def __init__(self, total_iteration: int, origin_model: Module, origin_config_list: List[Dict],
                 pruning_algorithm: str, evaluator: Callable[[Module], float],
                 origin_masks: Dict[str, Dict[str, Tensor]] = {}, start_temperature: float = 100,
                 stop_temperature: float = 20, cool_down_rate: float = 0.9, perturbation_magnitude: float = 0.35,
                 log_dir: str = '.', keep_intermediate_result: bool = False):
        self.iterative_pruner = SimulatedAnnealingPruner(model=None,
                                                         config_list=None,
                                                         pruning_algorithm=pruning_algorithm,
                                                         evaluator=evaluator,
                                                         start_temperature=start_temperature,
                                                         stop_temperature=stop_temperature,
                                                         cool_down_rate=cool_down_rate,
                                                         perturbation_magnitude=perturbation_magnitude,
                                                         log_dir=Path(log_dir, 'SA'),
                                                         keep_intermediate_result=False)
        super().__init__(total_iteration=total_iteration,
                         origin_model=origin_model,
                         origin_config_list=origin_config_list,
                         origin_masks=origin_masks,
                         log_dir=log_dir,
                         keep_intermediate_result=keep_intermediate_result)

    def _iterative_pruner_reset(self, model: Module, config_list: List[Dict] = [], masks: Dict[str, Dict[str, Tensor]] = {}):
        self.iterative_pruner.task_generator._log_dir = Path(self._log_dir_root, 'SA')
        self.iterative_pruner.reset(model, config_list=config_list, masks=masks)

    def allocate_sparsity(self, new_config_list: List[Dict], model: Module, masks: Dict[str, Dict[str, Tensor]]):
        self._iterative_pruner_reset(model, new_config_list, masks)
        self.iterative_pruner.compress()
        _, _, _, _, config_list = self.iterative_pruner.get_best_result()
        return config_list


class AutoCompressPruner(IterativePruner):
    """
    Parameters
    ----------
    model : Module
        The origin unwrapped pytorch model to be pruned.
    config_list : List[Dict]
        The origin config list provided by the user. Note that this config_list is directly config the origin model.
        This means the sparsity provided by the origin_masks should also be recorded in the origin_config_list.
    total_iteration : int
        The total iteration number.
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
    admm_params : Dict
        If the pruner corresponding to the chosen pruning_algorithm has extra parameters, put them as a dict to pass in.
    """

    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int, evaluator: Callable[[Module], float],
                 start_temperature: float = 100, stop_temperature: float = 20, cool_down_rate: float = 0.9,
                 perturbation_magnitude: float = 0.35, log_dir: str = '.', keep_intermediate_result: bool = False,
                 finetuner: Optional[Callable[[Module], None]] = None, speed_up: bool = False, dummy_input: Optional[Tensor] = None,
                 admm_params: Dict = {}):
        task_generator = AutoCompressTaskGenerator(total_iteration=total_iteration,
                                                   origin_model=model,
                                                   origin_config_list=config_list,
                                                   pruning_algorithm='level',
                                                   evaluator=evaluator,
                                                   start_temperature=start_temperature,
                                                   stop_temperature=stop_temperature,
                                                   cool_down_rate=cool_down_rate,
                                                   perturbation_magnitude=perturbation_magnitude,
                                                   log_dir=log_dir,
                                                   keep_intermediate_result=keep_intermediate_result)
        pruner = ADMMPruner(None, None, **admm_params)
        super().__init__(pruner, task_generator, finetuner=finetuner, speed_up=speed_up, dummy_input=dummy_input,
                         evaluator=evaluator, reset_weight=False)
