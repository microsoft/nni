from copy import deepcopy
import functools
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

from torch import Tensor
from torch.nn import Module

from nni.compression.pytorch.compressor import Pruner
from nni.compression.pytorch.speedup import ModelSpeedup

from .sparsity_generator import SparsityGenerator, NaiveSparsityGenerrator


class PruningScheduler:
    def __init__(self, model: Module, config_list: List[Dict], iteration_num: Optional[int] = None,
                 sparsity_generator: Optional[SparsityGenerator] = None, trainer: Callable = None):
        self.origin_model = model
        self.origin_config_list = config_list
        self.iteration_num = iteration_num
        self.sparsity_generator = sparsity_generator if sparsity_generator else NaiveSparsityGenerrator(config_list, iteration_num)

        self.pruner_cls = None

    def reset(self):
        self.sparsity_generator.reset(self.origin_config_list, self.iteration_num)

    def set_pruner(self, pruner_cls: Callable[[Module, List[Dict]], Pruner], **pruner_kwargs):
        self.pruner_cls = functools.partial(pruner_cls, **pruner_kwargs)

    def compress(self, log_dir: str = './log', tag: str = 'default'):
        model = deepcopy(self.origin_model)
        config_list = self.sparsity_generator.generate_config_list(deepcopy(self.origin_model), deepcopy(self.origin_config_list))
        iteration_round = 0
        log_dir = Path(log_dir, '{}-{}'.format(tag, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))

        while config_list:
            iteration_round += 1
            log_path = Path(log_dir, iteration_round)
            model_path = Path(log_path, 'model.pth')
            mask_path = Path(log_path, 'mask.pth')

            # pruning
            pruner = self.pruner_cls(model, config_list)
            pruner.compress()
            pruner.export_model(model_path=model_path, mask_path=mask_path)

            # speed up
            pruner._unwrap_model()
            compact_model, mask = self.speed_up(model, mask_path)
            finetuner(compact_model)

            real_config_list = compute_sparsity(model, compact_model)

            config_list = self.sparsity_generator.generate_config_list(model, config_list)

        return compact_model, mask, real_config_list

    def speed_up(self, model: Module, mask_path: str, dummy_input: Tensor, device: str = None):
        ModelSpeedup(model, dummy_input, mask_path, map_location=device)
