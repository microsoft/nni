# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Optional, Tuple

from torch.nn import Module
from torch.tensor import Tensor


class PruningScheduler:
    def generate_task(self) -> Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]:
        """
        Returns
        -------
        Tuple[int, Module, List[Dict], Dict[str, Dict[str, Tensor]]]
            Return new task id, model under pruning, config list used in this task and pre-masks.
        """
        raise NotImplementedError()

    def record_task_result(self, task_id: int, pruned_model: Module, masks: Dict[str, Dict[str, Tensor]], score: float):
        """
        Used to record the task result.
        Parameters
        ----------
        task_id
            The id of the finished task.
        pruned_model
            The pruned model after `pruning_one_step`.
        masks
            The masks should be applied on the pruned model.
        score
            The score of the pruning performance in this task.
        """
        raise NotImplementedError()

    def pruning_one_step(self, model: Module, config_list: List[Dict], masks: Dict[str, Dict[str, Tensor]]) -> Tuple[Module, Dict[str, Dict[str, Tensor]], float]:
        """
        Pruning the model with config list.
        Parameters
        ----------
        model
            The model under pruning.
        config_list
            The config list usually identify the layers that want to prune.
        masks
            The masks should be applied on the under pruning model.
        """
        raise NotImplementedError()

    def get_best_result(self) -> Tuple[int, Module, Dict[str, Dict[str, Tensor]], float]:
        """
        Returns
        -------
        Tuple[int, Module, Dict[str, Dict[str, Tensor]], float]
            Return the task result that has the best performance,
            inculde task id, the pruned model, the masks on the pruned model and score.
        """
        raise NotImplementedError()

    def compress(self) -> Tuple[Module, Dict[str, Dict[str, Tensor]]]:
        """
        The pruning schedule main loop.

        Returns
        -------
        Tuple[Module, Dict[str, Dict[str, Tensor]]]
            Return the pruned_model and the masks on it in the last iteration.
        """
        task_id, model, config_list, pre_masks = self.generate_task()
        pruned_model, masks = None, None

        while task_id is not None:
            pruned_model, masks, score = self.pruning_one_step(model, config_list, pre_masks)
            self.record_task_result(task_id, pruned_model, masks, score)
            task_id, model, config_list, pre_masks = self.generate_task()

        return pruned_model, masks
