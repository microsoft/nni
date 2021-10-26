# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Dict, List, Callable

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.pruning.iterative_pruner import SimulatedAnnealingPruner

from .task_generator import LotteryTicketTaskGenerator

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
                                                         log_dir=Path(log_dir),
                                                         keep_intermediate_result=False)
        super().__init__(total_iteration=total_iteration,
                         origin_model=origin_model,
                         origin_config_list=origin_config_list,
                         origin_masks=origin_masks,
                         log_dir=log_dir,
                         keep_intermediate_result=keep_intermediate_result)

    def allocate_sparsity(self, new_config_list: List[Dict], model: Module, masks: Dict[str, Dict[str, Tensor]]):
        self.iterative_pruner.reset(model, new_config_list, masks)
        self.iterative_pruner.compress()
        _, _, _, _, config_list = self.get_best_result()
        return config_list
