from __future__ import annotations

from copy import deepcopy
import logging
from typing import Dict, List, Literal

import torch

from .wrapper import ModuleWrapper, register_wrapper

_logger = logging.getLogger(__name__)


class Compressor:
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], mode: Literal['pruning', 'quantization', 'distillation'],
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None):
        self.bound_model = model
        self.config_list = deepcopy(config_list)

        self._is_wrapped = False
        self._module_wrappers = register_wrapper(model, config_list, mode, existed_wrapper)

    @classmethod
    def _from_compressor(cls, compressor, new_config_list: List[Dict], mode: Literal['pruning', 'quantization', 'distillation']):
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
        pass
