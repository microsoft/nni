# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import logging
from typing import Dict, List, Literal

import torch

from nni.compression.pytorch.utils.scaling import Scaling

from .config import trans_legacy_config_list
from .target_space import TargetType, DistillationTargetSpace, PruningTargetSpace, QuantizationTargetSpace
from .wrapper import ModuleWrapper, register_wrappers
from ..utils.evaluator import Evaluator

_logger = logging.getLogger(__name__)


class Compressor:
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], mode: Literal['pruning', 'quantization', 'distillation'],
                 evaluator: Evaluator | None = None, existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        assert mode in ['pruning', 'quantization', 'distillation']
        self.bound_model = model
        self.config_list = trans_legacy_config_list(deepcopy(config_list))
        self._validate_config()
        self.evaluator = evaluator
        if self.evaluator is not None:
            assert isinstance(evaluator, Evaluator)
            if not evaluator._initialization_complete:
                evaluator._init_optimizer_helpers(self.bound_model)

        self._is_wrapped = False
        self._module_wrappers, self._target_spaces = register_wrappers(self.bound_model, self.config_list, mode, existed_wrapper)
        self.wrap_model()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], *args, **kwargs):
        """
        Inherited from the compressor exsited wrapper to initialize a new compressor.
        """
        if compressor._is_wrapped:
            compressor.unwrap_model()
        model = compressor.bound_model
        existed_wrapper = compressor._module_wrappers
        evaluator = compressor.evaluator
        # note that here don't have `mode` because subclass should know what its mode is.
        return cls(model=model, config_list=new_config_list, evaluator=evaluator, existed_wrapper=existed_wrapper, *args, **kwargs)

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

    def _get_param_names_map(self) -> Dict[str, str]:
        param_names_map = {}
        if self._is_wrapped is True:
            for param_name, _ in self.bound_model.named_parameters():
                origin_param_name = ''.join(param_name.split('_nni_wrapper.')) if '_nni_wrapper' in param_name else param_name
                param_names_map[origin_param_name] = param_name
        else:
            raise RuntimeError('Only can get param_names_map when the model is wrapped.')
        return param_names_map

    def compress(self):
        raise NotImplementedError()


class Pruner(Compressor):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        super().__init__(model=model, config_list=config_list, mode='pruning', evaluator=evaluator,
                         existed_wrapper=existed_wrapper, *args, **kwargs)
        self._register_scalers()
        self._target_spaces: Dict[str, Dict[str, PruningTargetSpace]]

    def _register_scalers(self):
        # scalers are used to support different sparse granularity
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.sparse_granularity is None:
                    continue
                if target_space.sparse_granularity == 'default':
                    target_space.sparse_granularity = self._set_default_sparse_granularity(target_space)
                if target_space.sparse_granularity == 'out_channel':
                    assert target_space._target_type is TargetType.PARAMETER
                    target_space._scaler = Scaling([1], kernel_padding_mode='back', kernel_padding_val=-1)
                elif target_space.sparse_granularity == 'in_channel':
                    assert target_space._target_type is TargetType.PARAMETER
                    target_space._scaler = Scaling([1], kernel_padding_mode='front', kernel_padding_val=-1)
                else:
                    assert all(isinstance(_, int) for _ in target_space.sparse_granularity)
                    target_space._scaler = Scaling(target_space.sparse_granularity, kernel_padding_mode='front', kernel_padding_val=1)

    def _set_default_sparse_granularity(self, target_space: PruningTargetSpace) -> List[int] | str | None:
        return 'out_channel'

    def get_masks(self) -> Dict[str, Dict[str, torch.Tensor]]:
        masks = defaultdict(dict)
        for module_name, wrapper in self._module_wrappers.items():
            for target_name, target_space in wrapper.pruning_target_spaces.items():
                masks[module_name][target_name] = target_space.mask.clone()
        return masks

    def update_masks(self, masks: Dict[str, Dict[str, torch.Tensor]]):
        for module_name, target_masks in masks.items():
            assert module_name in self._module_wrappers, f'{module_name} is not register in this compressor, can not update mask for it.'
            wrapper = self._module_wrappers[module_name]
            for target_name, target_mask in target_masks.items():
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
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        super().__init__(model=model, config_list=config_list, mode='quantization', evaluator=evaluator,
                         existed_wrapper=existed_wrapper, *args, **kwargs)
        self._target_spaces: Dict[str, Dict[str, QuantizationTargetSpace]]


class Distiller(Compressor):
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None,
                 existed_wrapper: Dict[str, ModuleWrapper] | None = None, *args, **kwargs):
        super().__init__(model=model, config_list=config_list, mode='distillation', evaluator=evaluator,
                         existed_wrapper=existed_wrapper, *args, **kwargs)
        self._target_spaces: Dict[str, Dict[str, DistillationTargetSpace]]
