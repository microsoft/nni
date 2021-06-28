from copy import deepcopy
import functools
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

import json_tricks
from torch import Tensor
from torch.nn import Module

from nni.compression.pytorch import apply_compression_results
from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.speedup import ModelSpeedup

from sparsity_generator import SparsityGenerator, NaiveSparsityGenerrator
from utils import compute_sparsity_with_compact_model, compute_sparsity_with_mask

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PruningScheduler:
    def __init__(self, model: Module, config_list: List[Dict], iteration_num: Optional[int] = None,
                 sparsity_generator: Optional[SparsityGenerator] = None):
        self.origin_model = model
        self.origin_config_list = config_list
        self.sparsity_generator = sparsity_generator if sparsity_generator else NaiveSparsityGenerrator(config_list, iteration_num)

        self.pruner_cls = None

    def reset(self):
        self.sparsity_generator.reset(self.origin_config_list, self.iteration_num)

    def set_pruner(self, pruner_cls: Callable[[Module, List[Dict]], Pruner], **pruner_kwargs):
        self.pruner_cls = functools.partial(pruner_cls, **pruner_kwargs)

    def compress(self, finetuner: Callable = None, finetune_optimizer_gen: Callable = None, finetune_dataloader=None,
                 finetune_epochs: int = None, speed_up: bool = False, dummy_input: Tensor = None,
                 log_dir: str = './log', tag: str = 'default', consistent: bool = True):
        model = deepcopy(self.origin_model)
        config_list = self.sparsity_generator.generate_config_list(deepcopy(self.origin_model), deepcopy(self.origin_config_list))
        iteration_round = 0
        log_dir = Path(log_dir, '{}-{}'.format(tag, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))

        logger.info('Pruning start...')
        while config_list:
            iteration_round += 1
            logger.info('###### Pruning Iteration %i ######', iteration_round)
            logger.info('config list in this iteration:\n%s', json_tricks.dumps(config_list, indent=4))

            log_path = Path(log_dir, str(iteration_round))
            log_path.mkdir(parents=True, exist_ok=True)
            model_path = Path(log_path, 'model.pth')
            mask_path = Path(log_path, 'mask.pth')
            model = model if consistent else deepcopy(self.origin_model)

            # pruning
            pruner = self.pruner_cls(model, config_list)
            pruner.compress()
            pruner.export_model(model_path=model_path, mask_path=mask_path)
            pruner.get_pruned_weights()

            # speed up
            if speed_up:
                assert dummy_input is not None
                pruner._unwrap_model()
                masked_model = deepcopy(model)
                self.speed_up(model, mask_path, dummy_input)

            # fine-tuning
            if finetuner:
                assert finetune_optimizer_gen is not None and finetune_dataloader is not None
                optimizer = finetune_optimizer_gen(model)
                for i in range(finetune_epochs):
                    finetuner(model, optimizer, finetune_dataloader, i)

            # compute real sparsity
            if speed_up:
                real_config_list = compute_sparsity_with_compact_model(masked_model, model, config_list)
            else:
                real_config_list = compute_sparsity_with_mask(model, mask_path, config_list)
                pruner._unwrap_model()
                apply_compression_results(model, mask_path)

            config_list = self.sparsity_generator.generate_config_list(model, real_config_list)

        logger.info('Pruning end.')
        return model

    def speed_up(self, model: Module, mask_path: str, dummy_input: Tensor, device: str = None):
        ms = ModelSpeedup(model, dummy_input, mask_path, map_location=device)
        ms.speedup_model()

    def get_best_config_list(self):
        return self.sparsity_generator.best_config_list
