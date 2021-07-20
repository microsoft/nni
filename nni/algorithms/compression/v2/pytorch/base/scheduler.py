# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

from torch.nn import Module
from torch.tensor import Tensor

from nni.compression.pytorch.speedup import ModelSpeedup
from nni.algorithms.compression.v2.pytorch.utils import apply_compression_results, compute_sparsity_with_compact_model, compute_sparsity_with_mask
from .pruner import Pruner

_logger = logging.getLogger(__name__)


class ConfigGenerator:
    """
    This class used to generate config list for pruner in each iteration.
    """
    def __init__(self, model: Module, config_list: List[Dict]):
        self.origin_model = model
        self.origin_config_list = config_list
        self._best_pruned_model = None
        self._best_config_list = None

    @property
    def best_pruned_model(self) -> Optional(Module):
        """
        Return the pruned model with the highest score.
        """
        return self._best_pruned_model

    @property
    def best_config_list(self) -> Optional(List[Dict]):
        """
        Return the generated config with the highest score.
        """
        return self._best_config_list

    def generate_config_list(self, iteration: int) -> Optional[Tuple[Module, List[Dict], Optional[Dict[str, Dict[str, Tensor]]]]]:
        """
        Parameters
        ----------
        The current iteration.

        Returns
        -------
        Optional[Tuple[Module, List[Dict], Optional[Dict[str, Dict[str, Tensor]]]]]
            (model, config_list, masks).
            The model to compress, config list and masks for this iteration, None means no more iterations.
        """
        raise NotImplementedError()

    def receive_compression_result(self, model: Module, score: Optional[float],
                                   real_sparsity_config_list: List[Dict], masks: Optional[Dict[str, Tensor]]):
        """
        Parameters
        ----------
        model
            The pruned model in the last iteration. It might be a sparsify model or a speed-up model.
        score
            The score of the model, higher score means better performance.
        real_sparsity_config_list
            Real sparsity config list.
        masks
            If masks is None, the pruned model is a compact model after speed up.
            If masks is not None, the pruned model is a sparsify model without speed up.
        """
        raise NotImplementedError()


class PruningScheduler:
    def __init__(self, pruner: Pruner, config_generator: ConfigGenerator, evaluator: Callable[[Module], float],
                 finetuner: Callable[[Module]] = None, speed_up: bool = False, dummy_input: Tensor = None,
                 log_dir: str = '.'):
        """
        Parameters
        ----------
        pruner
            The pruner used in pruner scheduler.
            The scheduler will use `Pruner.reset(model, config_list)` to reset it in each iteration.
        config_generator
            Used to generate config_list for each iteration.
        evaluator
            Evaluate the pruned model and give a score.
        finetuner
            The finetuner handled all finetune logic, use a pytorch module as input.
        speed_up
            If set True, speed up the model in each iteration.
        dummy_input
            If `speed_up` is True, `dummy_input` is required for trace the model in speed up.
        """
        self.pruner = pruner
        self.config_generator = config_generator
        self.evaluator = evaluator
        self.finetuner = finetuner
        self.speed_up = speed_up
        self.dummy_input = dummy_input

        self.log_dir = Path(log_dir)

    def compress_one_step(self, model: Module, config_list: List[Dict], masks: Optional[Dict[str, Dict[str, Tensor]]],
                          log_dir: str) -> Tuple[Module, List[Dict]]:
        # compress model and export mask
        self.pruner.reset(model, config_list)
        if masks is not None:
            self.pruner.load_masks(masks)
        model, masks = self.pruner.compress()
        self.pruner.show_pruned_weights()
        self.pruner.export_model(model_path=Path(log_dir, 'pruned_model.pth'), mask_path=Path(log_dir, 'pruned_model.pth'))

        # apply masks to sparsify model
        self.pruner._unwrap_model()
        apply_compression_results(model, masks)

        # speed up and compute real sparsity
        if self.speed_up:
            origin_structure_model = deepcopy(model)
            ModelSpeedup(model, self.dummy_input, Path(log_dir, 'pruned_model.pth')).speedup_model()
            real_config_list = compute_sparsity_with_compact_model(origin_structure_model, model, config_list)
            masks = None
        else:
            real_config_list = compute_sparsity_with_mask(model, masks, config_list)

        # finetune
        if self.finetuner is not None:
            if self.speed_up:
                self.finetuner(model)
            else:
                self.pruner._wrap_model()
                self.finetuner(model)
                self.pruner._unwrap_model()

        # evaluate
        score = self.evaluator(model)

        return model, score, real_config_list, masks

    def compress(self):
        iteration = 0
        model, config_list, masks = self.config_generator.generate_config_list(iteration)

        while config_list is not None:
            log_dir = Path(self.log_dir, str(iteration))
            pruned_model, score, real_config_list, masks = self.compress_one_step(model, config_list, masks, log_dir)
            _logger.info('\nIteration %d\nscore: %f\nconfig list:\n%s', iteration, score, real_config_list)

            iteration += 1
            self.config_generator.receive_compression_result(pruned_model, score, real_config_list, masks)
            model, config_list, masks = self.config_generator.generate_config_list(iteration)

        return self.config_generator.best_pruned_model
