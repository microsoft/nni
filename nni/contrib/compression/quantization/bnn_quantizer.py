# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..base.target_space import TargetType
from ..utils.evaluator import Evaluator


class BNNQuantizer(Quantizer):
    '''
    BinaryNet Quantization: https://arxiv.org/pdf/1602.02830.pdf
    '''
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None, \
                 fused_module_lis: List[List[str]] = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers, \
                         fused_module_lis=fused_module_lis)

        self.is_init = False
        self.register_bnn_apply_method()
        self.register_track_func()

    def _validate_config(self):
        '''
        Only support to quantize weight and output
        '''
        pass

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale_zp)

    def register_bnn_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                target_space.apply_method = 'bnn_clamp_round'

    def init_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init:
            return
        if target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        target_space.zero_point = torch.tensor(0.0).to(target.device)
        target_space.scale = torch.tensor(1.0).to(target.device)

    def register_trigger(self, evaluator: Evaluator):
        def optimizer_task():
            self.is_init = True
            # clip params to (-1,1)
            for _, ts in self._target_spaces.items():
                for target_name, target_space in ts.items():
                    if target_space.type is TargetType.PARAMETER and 'weight' in target_name:
                        target_space.target.data = target_space.target.data.clamp(-1,1)

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map(), self.patch_optimizer_param_group())
        self.register_trigger(self.evaluator)
        self.evaluator.train(max_steps, max_epochs)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_calibration_config()

    def get_calibration_config(self):
        return None