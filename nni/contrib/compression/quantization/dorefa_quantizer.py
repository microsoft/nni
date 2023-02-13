# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..base.setting import INPUT_PREFIX, OUTPUT_PREFIX
from ..utils.evaluator import Evaluator


class DoReFaQuantizer(Quantizer):
    '''
    Dorefa-Quantizer: https://arxiv.org/pdf/1606.06160.pdf
    '''
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator | None = None, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None, \
                 fused_module_lis: List[List[str]] = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers, \
                         fused_module_lis=fused_module_lis)

        self.register_dorefa_apply_method()
        self.register_track_func()


    def _validate_config(self):
        '''
        1. only support quant-deqaunt weight and input
        2. quant-dequant output means quant the gradient of output 
        '''
        pass

    def register_dorefa_apply_method(self):
        for _, ts in self._target_spaces.items():
            for target_name, target_space in ts.items():
                if 'weight' in target_name:
                    target_space.apply_method = 'dofera_clamp_round_weight'

                elif INPUT_PREFIX in target_name:
                    target_space.apply_method = "dofera_clamp_round_input"
                elif OUTPUT_PREFIX in target_name:
                    target_space.apply_method = "dofera_clamp_round_output"

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.update_scale_zp)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]

        if INPUT_PREFIX in target_name or "weight" in target_name: #zero_point and scale don't change anymore
            tracked_max = torch.tensor(1.0).to(target.device)
            tracked_min = torch.tensor(0.0).to(target.device)
            scale, zero_point = update_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')
        elif OUTPUT_PREFIX in target_name:
            tracked_max = torch.tensor(1.0 + 0.5 / (2**target_space.quant_bits - 1)).to(target.device)
            tracked_min = torch.tensor(0 - 0.5 / (2**target_space.quant_bits - 1)).to(target.device)
            scale, zero_point = update_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')

        target_space.scale, target_space.zero_point = scale, zero_point

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map(), self.patch_optimizer_param_group())
        self.evaluator.train(max_steps, max_epochs)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_calibration_config()

    def get_calibration_config(self):
        pass


def update_scale_zp(tracked_max: Tensor, tracked_min: Tensor, qmax: int, qmin: int, quant_scheme: str = None):
    tracked_min = torch.min(tracked_min, torch.zeros_like(tracked_min))
    tracked_max = torch.max(tracked_max, torch.zeros_like(tracked_max))
    zero_point = torch.zeros_like(tracked_min)
    if quant_scheme == 'affine':
        scale = (tracked_max - tracked_min) / float(qmax - qmin)
        scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
        zero_point = qmin - torch.round(tracked_min / scale)
    elif quant_scheme in ['symmetric', None]:
        raise ValueError(f"Unsupported quant_scheme {quant_scheme}")
    else:
        raise RuntimeError(f'Unknown quant_scheme {quant_scheme}')

    zero_point = torch.clamp(zero_point, qmin, qmax)
    return scale, zero_point
