# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
from typing import List, Dict, Union

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator


class PtqQuantizer(Quantizer):
    '''
    Post Training Quantization
    '''
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_compressed = False
        self.register_ptq_apply_method()
        self.register_track_func()

    def _validate_config(self):
        pass

    def register_ptq_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                target_space.apply_method = 'clamp_round' if self.is_compressed else 'bypass'

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.track_min_max_val)

    def track_min_max_val(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_compressed:
            return
        def amin_reduce_func(converted_target: Tensor):
            return converted_target.detach().amin(dim=-1)

        def amax_reduce_func(converted_target: Tensor):
            return converted_target.detach().amax(dim=-1)

        if target_name not in wrapper.quantization_target_spaces:
            return

        target_space = wrapper.quantization_target_spaces[target_name]
        # TODO sync the collection of data info when using ddp
        if target_space._scaler:
            current_amin = target_space._scaler.shrink(target, amin_reduce_func, keepdim=True)
            current_amax = target_space._scaler.shrink(target, amax_reduce_func, keepdim=True)
        else:
            current_amin = target.detach().reshape(-1).amin(-1)
            current_amax = target.detach().reshape(-1).amax(-1)

        # update
        target_space.tracked_max = update_tracked_value(target_space.tracked_max, current_amax, "max")
        target_space.tracked_min = update_tracked_value(target_space.tracked_min, current_amin, "min")

    def update_scale_zp(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.tracked_max is None or target_space.tracked_min is None:
                    return
                tracked_min = torch.min(target_space.tracked_min, torch.zeros_like(target_space.tracked_min))
                tracked_max = torch.max(target_space.tracked_max, torch.zeros_like(target_space.tracked_max))
                zero_point = torch.zeros_like(tracked_min)
                if target_space.quant_scheme in ['symmetric', None]:
                    abs_max = torch.max(torch.abs(tracked_min), torch.abs(tracked_max))
                    scale = abs_max / (float(target_space.qmax - target_space.qmin) / 2)
                    scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                    # NOTE: here need to check, +1 because in pytorch, symmetric qint8 zp is 0, quint8 zp is 128.
                    zero_point_val = (target_space.qmax + target_space.qmin + 1) // 2
                    zero_point = torch.full_like(zero_point, zero_point_val)
                elif target_space.quant_scheme == 'affine':
                    scale = (tracked_max - tracked_min) / float(target_space.qmax - target_space.qmin)
                    scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                    zero_point = target_space.qmin - torch.round(tracked_min / scale)
                else:
                    raise RuntimeError(f'Unknown quant_scheme {target_space.quant_scheme}')
                zero_point = torch.clamp(zero_point, target_space.qmin, target_space.qmax)
                target_space.scale, target_space.zero_point = scale, zero_point

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        evaluator.patch_optim_param_group(self.patch_optimizer_param_group())

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        self.evaluator.evaluate()
        # compute and update scale and zero
        self.update_scale_zp()
        self.is_compressed = True
        self.register_ptq_apply_method()


def update_tracked_value(original_val: Union[Tensor, None], current_val: Tensor, mode: str="max"):
    if original_val is None:
        return current_val
    assert current_val is not None
    assert original_val.shape == current_val.shape
    if mode == "max":
        return torch.max(original_val, current_val)
    elif mode == "min":
        return torch.min(original_val, current_val)
    else:
        raise TypeError(f"don't support type:{mode}")
