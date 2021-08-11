# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Callable, Optional

from torch import Tensor
from torch.nn import Module

from nni.algorithms.compression.v2.pytorch.base.scheduler import PruningScheduler
from nni.algorithms.compression.v2.pytorch.common.consistent_task_generator import AGPTaskGenerator
from .basic_pruner import LevelPruner


class AGPPruning:
    def __init__(self, model: Module, config_list: List[Dict], total_iteration: int,
                 finetuner: Callable[[Module], None],
                 masks: Dict[str, Dict[str, Tensor]] = {}, log_dir: str = './log',
                 speed_up: bool = False, dummy_input: Tensor = None,
                 evaluator: Optional[Callable[[Module], float]] = None):
        """
        Parameters
        ----------
        model
            The model under pruning.
        config_list
            Supported keys:
                - sparsity : This is to specify the sparsity operations to be compressed to.
                - op_types : Operation module types to prune.
                - op_names : Operation module names to prune.
                - exclude : If `exclude` is seted as True, `op_types` and `op_names` seted in this config will be excluded from pruning.
        total_iteration
            
        """
        # NOTE: model used by task generator must be a unwrapped model.
        task_generator = AGPTaskGenerator(total_iteration, model, config_list, masks, log_dir)
        pruner = LevelPruner(model, config_list)
        self.scheduler = PruningScheduler(pruner, task_generator, finetuner, speed_up, dummy_input, evaluator)

    def compress(self) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        return self.scheduler.compress()

    def get_best_result(self) -> Optional[Tuple[Module, Dict[str, Dict[str, Tensor]]]]:
        return self.scheduler.get_best_result()
