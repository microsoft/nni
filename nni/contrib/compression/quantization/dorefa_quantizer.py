# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict, Union

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
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.register_dorefa_apply_method()
        self.register_track_func()

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
        if not wrapper.training or target_name not in wrapper.quantization_target_spaces:
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
        else:
            raise RuntimeError(f'Unknown target_name {target_name}')

        target_space.scale, target_space.zero_point = scale, zero_point

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        evaluator.patch_optim_param_group(self.patch_optimizer_param_group())

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        pass


def update_scale_zp(tracked_max: Tensor, tracked_min: Tensor, qmax: int, qmin: int, quant_scheme: Union[str, None] = None):
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
