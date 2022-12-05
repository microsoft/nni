from __future__ import annotations

from copy import deepcopy
import logging
from typing import Dict, List, Literal

import torch

from .config import trans_legacy_config_list
from .wrapper import ModuleWrapper, register_wrappers

_logger = logging.getLogger(__name__)


class Compressor:
    def __init__(self, model: torch.nn.Module, config_list: List[Dict],
                 mode: Literal['pruning', 'quantization', 'distillation'],
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None):
        self.bound_model = model
        self.config_list = trans_legacy_config_list(deepcopy(config_list))

        self._is_wrapped = False
        self._module_wrappers = register_wrappers(model, config_list, mode, existed_wrapper)

        self.wrap_model()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict],
                        mode: Literal['pruning', 'quantization', 'distillation']):
        """
        Inherited from the compressor exsited wrapper to initialize a new compressor.
        """
        if compressor._is_wrapped:
            compressor.unwrap_model()
        model = compressor.bound_model
        existed_wrapper = compressor._module_wrappers
        return cls(model, new_config_list, mode, existed_wrapper)

    def _validate_config(self):
        pass

    def wrap_model(self):
        if self._is_wrapped is True:
            warn_msg = 'The bound model has been wrapped, no need to wrap again.'
            _logger.warning(warn_msg)
        for _, wrapper in self._module_wrappers.items():
            wrapper.wrap()
        self._is_wrapped = True

    def unwrap_model(self):
        if self._is_wrapped is False:
            warn_msg = 'The bounde model is not wrapped, can not unwrap it.'
            _logger.warning(warn_msg)
        for _, wrapper in self._module_wrappers.items():
            wrapper.unwrap()
        self._is_wrapped = False

    def compress(self):
        return self.bound_model


class Pruner(Compressor):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        super().__init__(model, config_list, mode='pruning')

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict]):
        return super().from_compressor(compressor, new_config_list, mode='pruning')

    def update_masks(self, masks: Dict[str, Dict[str, torch.Tensor]]):
        for module_name, target_masks in masks.items():
            assert module_name in self._module_wrappers, f'{module_name} is not register in this compressor, can not update mask for it.'
            wrapper = self._module_wrappers[module_name]
            for target_name, target_mask in target_masks:
                assert target_name in wrapper.pruning_target_spaces, \
                    f'{module_name}.{target_name} is not a pruning target, can not update mask for it.'
                wrapper.pruning_target_spaces[target_name].mask = target_mask

    def _collect_data(self) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def _calculate_metrics(self, data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def _generate_sparsity(self, metrics: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def generate_masks(self) -> Dict[str, Dict[str, torch.Tensor]]:
        data = self._collect_data()
        metrics = self._calculate_metrics(data)
        return self._generate_sparsity(metrics)

    def compress(self):
        masks = self.generate_masks()
        self.update_masks(masks)
        return self.bound_model, masks


class Quantizer(Compressor):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        super().__init__(model, config_list, mode='quantization')

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict]):
        return super().from_compressor(compressor, new_config_list, mode='quantization')


class Distiller(Compressor):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict]):
        super().__init__(model, config_list, mode='distillation')

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict]):
        return super().from_compressor(compressor, new_config_list, mode='distillation')
