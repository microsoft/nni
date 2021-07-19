from copy import deepcopy
from io import SEEK_CUR
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from numpy import real

import torch
from torch.nn import Module
from torch.tensor import Tensor

from nni.compression.pytorch.speedup import ModelSpeedup
from .pruner import Pruner

_logger = logging.getLogger(__name__)


class ConfigGenerator:
    """
    This class used to generate config list for pruner in each iteration.
    """
    def __init__(self, config_list: List[Dict]):
        pass

    @property
    def best_config_list(self) -> Optional(List[Dict]):
        raise NotImplementedError()

    def generate_config_list(self, model: Module, real_sparsity_config_list: List[Dict]) -> Optional[List[Dict]]:
        """
        Parameters
        ----------
        model
            The model that wants to sparsify.
        real_sparsity_config_list
            Real sparsity config list.

        Returns
        -------
        Optional[List[Dict]]
            The config list for this iteration, None means no more iterations.
        """
        raise NotImplementedError()

    def reset(self, origin_config_list: List[Dict], iteration_num: Optional[int] = None):
        """
        Parameters
        ----------
        origin_config_list
            The origin config list.
        """
        raise NotImplementedError()


class PruningScheduler:
    def __init__(self, pruner: Pruner, config_generator: ConfigGenerator, finetuner: Callable[[Module]] = None,
                 speed_up: bool = False, dummy_input: Tensor = None, consistant: bool = False):
        self.pruner = pruner
        self.config_generator = config_generator
        self.finetuner = finetuner
        self.speed_up = speed_up
        self.dummy_input = dummy_input

        self.consistant = consistant

    def compute_sparsity_with_compact_model(self, origin_model: Module, compact_model: Module, config_list: List[Dict]) -> List[Dict]:
        real_config_list = []
        for config in config_list:
            left_weight_num = 0
            total_weight_num = 0
            for module_name, module in origin_model.named_modules():
                module_type = type(module).__name__
                if 'op_types' in config and module_type not in config['op_types']:
                    continue
                if 'op_names' in config and module_name not in config['op_names']:
                    continue
                total_weight_num += module.weight.data.numel()
            for module_name, module in compact_model.named_modules():
                module_type = type(module).__name__
                if 'op_types' in config and module_type not in config['op_types']:
                    continue
                if 'op_names' in config and module_name not in config['op_names']:
                    continue
                left_weight_num += module.weight.data.numel()
            real_config_list.append(deepcopy(config_list))
            real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
        return real_config_list

    def compute_sparsity_with_mask(self, masked_model: Module, masks: Dict[str, Tensor], config_list: List[Dict], dim: int = 0):
        real_config_list = []
        for config in config_list:
            left_weight_num = 0
            total_weight_num = 0
            for module_name, module in masked_model.named_modules():
                module_type = type(module).__name__
                if 'op_types' in config and module_type not in config['op_types']:
                    continue
                if 'op_names' in config and module_name not in config['op_names']:
                    continue
                weight_mask = masks[module_name]['weight']
                mask_size = weight_mask.size()
                if len(mask_size) == 1:
                    index = torch.nonzero(weight_mask.abs() != 0).tolist()
                else:
                    sum_idx = list(range(len(mask_size)))
                    sum_idx.remove(dim)
                    index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0).tolist()
                module_weight_num = module.weight.data.numel()
                left_weight_num += module_weight_num * len(index) / weight_mask.size(dim)
                total_weight_num += module_weight_num
            real_config_list.append(deepcopy(config_list))
            real_config_list[-1]['sparsity'] = 1 - left_weight_num / total_weight_num
        return real_config_list

    def compress_one_step(self, model: Module, config_list: List[Dict], log_dir: str):
        model, mask = self.pruner.compress()
        self.pruner.show_pruned_weights()
        self.pruner.export_model(model_path=Path(log_dir, 'pruned_model.pth'), mask_path=Path(log_dir, 'pruned_model.pth'))

        if self.speed_up:
            self.pruner._unwrap_model()
            origin_structure_model = deepcopy(model)
            ModelSpeedup(model, self.dummy_input, Path(log_dir, 'pruned_model.pth')).speedup_model()

        if self.finetuner is not None:
            self.finetuner(model)

        if self.speed_up:
            real_config_list = self.compute_sparsity_with_compact_model(origin_structure_model, model, config_list)
        else:
            self.pruner._unwrap_model()
            real_config_list = self.compute_sparsity_with_mask(model, mask, config_list)

        return real_config_list, model

    def compress(self):
        if self.consistant:
            model = self.pruner.bound_model

        model = deepcopy(self.origin_model)
        config_list = self.sparsity_generator.generate_config_list(deepcopy(self.origin_model), deepcopy(self.origin_config_list))
        iteration_round = 0
        log_dir = Path(log_dir, '{}-{}'.format(tag, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))

        _logger.info('Pruning start...')
        while config_list:
            iteration_round += 1
            _logger.info('###### Pruning Iteration %i ######', iteration_round)
            _logger.info('config list in this iteration:\n%s', json_tricks.dumps(config_list, indent=4))

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

        _logger.info('Pruning end.')
        return model

    def get_best_config_list(self):
        return self.config_generator.best_config_list
