# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator


class LsqQuantizer(Quantizer):
    '''
    LsqQuantizer: https://arxiv.org/pdf/1902.08153.pdf
    '''
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None, \
                 fused_module_lis: List[List[str]] = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers, \
                         fused_module_lis=fused_module_lis)

        self.is_init = False

        self.register_scale()
        self.register_lsq_apply_method()
        self.register_track_func()

    def _validate_config(self):
        '''
        1. for activation, only supports for uint, for weight, only supports for int 
        2. no zero_point
        '''
        pass

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale)

    def init_scale(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init:
            return
        if target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        init_target = target.data.detach().abs().mean() * 2 / (target_space.qmax ** 0.5)
        target_space.scale.data = init_target
        target_space.zero_point = torch.tensor(0.0).to(target.device)

    def register_lsq_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                target_space.apply_method = "lsq_clamp_round"

    def register_scale(self):
        for module_name, ts in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            for target_name, target_space in ts.items():
                if hasattr(wrapper, f"{target_name}_scale"):
                    delattr(wrapper, f"{target_name}_scale")
                try:
                    device = next(wrapper.parameters()).device
                except StopIteration:
                    try:
                        device = next(wrapper.buffers()).device
                    except StopIteration:
                        # NOTE: this will have risk in model parallel
                        device = next(self.bound_model.parameters()).device
                param = torch.nn.Parameter(torch.Tensor([1.0]))
                wrapper.register_parameter(f"{target_name}_scale", param)
                target_space.scale.data = target_space.scale.data.to(device)

    def patch_optimizer_param_group(self):
        module_name_param_dict = {}
        for module_name, ts in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            scale_param_lis = []
            if getattr(wrapper.module, "original_bias", None) is not None:
                scale_param_lis.append(wrapper.module.original_bias)
            for _, target_space in ts.items():
                scale_param_lis.append(target_space.scale)
            module_name_param_dict[module_name] = scale_param_lis

        return module_name_param_dict if len(module_name_param_dict) > 0 else None

    def register_trigger(self, evaluator: Evaluator):
        def optimizer_task():
            self.is_init = True

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map(), self.patch_optimizer_param_group())
        self.register_trigger(self.evaluator)
        self.evaluator.train(max_steps, max_epochs)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_calibration_config()

    def get_calibration_config(self):
        pass
