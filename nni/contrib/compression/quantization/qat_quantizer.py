# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Dict, List, overload

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator


class QATQuantizer(Quantizer):
    """
    TODO
    """
    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 quant_start_step: int = 0):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 quant_start_step: int = 0, existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 quant_start_step: int = 0, existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator
        self.quant_start_step = max(quant_start_step, 0)

        # scale and zero point will be computed during training when self.current_step >= self.quant_start_step
        self.current_step = 0
        self.register_qat_apply_method()
        self.register_track_func()

    def register_qat_apply_method(self):
        if self.current_step < self.quant_start_step:
            for _, ts in self._target_spaces.items():
                for _, target_space in ts.items():
                    target_space.apply_method = 'bypass'
        else:
            for _, ts in self._target_spaces.items():
                for _, target_space in ts.items():
                    target_space.apply_method = 'qat_clamp_round'

    def register_track_func(self):
        # NOTE: tracked min max value will be registered as buffer after the first forward during training,
        # scale and zero point will be registered as buffer after self.current_step >= self.quant_start_step.
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.track_min_max_val)
            wrapper.register_track_func(self.update_scale_zp)

    def track_min_max_val(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        # in a fused compression pipeline, the target name may be another compressor's target name
        if not wrapper.training or target_name not in self._target_spaces:
            return
        return track_min_max_val(wrapper, target_name, target)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if not wrapper.training or self.current_step < self.quant_start_step:
            return
        if target_name in wrapper.quantization_target_spaces:
            target_space = wrapper.quantization_target_spaces[target_name]
            if target_space.tracked_max is None or target_space.tracked_min is None:
                return
            # Comments copied from the old version:
            # Extend the [min, max] interval to ensure that it contains 0.
            # Otherwise, we would not meet the requirement that 0 be an exactly representable value.
            # I think this is for activations that need to be pad in the training.
            # However this is a default behavior in PyTorch quantization observer.
            # So we also make it a default behavior.
            tracked_min = torch.min(target_space.tracked_min, torch.zeros_like(target_space.tracked_min))
            tracked_max = torch.max(target_space.tracked_max, torch.zeros_like(target_space.tracked_max))
            zero_point = torch.zeros_like(tracked_min)
            qmin, qmax = target_space.qmin, target_space.qmax
            assert isinstance(qmin, int) and isinstance(qmax, int)
            if target_space.quant_scheme in ['symmetric', None]:
                abs_max = torch.max(torch.abs(tracked_min), torch.abs(tracked_max))
                scale = abs_max / (float(qmax - qmin) / 2)
                scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                # NOTE: here need to check, +1 because in pytorch, symmetric qint8 zp is 0, quint8 zp is 128.
                zero_point_val = (qmax + qmin + 1) // 2
                zero_point = torch.full_like(zero_point, zero_point_val)
            elif target_space.quant_scheme == 'affine':
                scale = (tracked_max - tracked_min) / float(qmax - qmin)
                scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                zero_point = qmin - torch.round(tracked_min / scale)
            else:
                raise RuntimeError(f'Unknown quant_scheme {target_space.quant_scheme}')
            zero_point = torch.clamp(zero_point, qmin, qmax)
            target_space.scale, target_space.zero_point = scale, zero_point

    def track_forward(self, *args, **kwargs):
        super().track_forward(*args, **kwargs)
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.scale is None:
                    assert target_space.tracked_max is not None
                    target_space.scale = torch.empty_like(target_space.tracked_max)
                if target_space.zero_point is None:
                    assert target_space.tracked_max is not None
                    target_space.zero_point = torch.empty_like(target_space.tracked_max)

    def register_trigger(self, evaluator: Evaluator):

        def optimizer_task():
            self.current_step += 1
            if self.current_step == self.quant_start_step:
                self.register_qat_apply_method()

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def compress(self, max_steps: int | None = None, max_epochs: int | None = None):
        self.evaluator.bind_model(self.bound_model, self._get_param_names_map())
        self.register_trigger(self.evaluator)
        self.evaluator.train(max_steps, max_epochs)
        self.evaluator.unbind_model()
        return self.bound_model, self.get_calibration_config()


def track_min_max_val(wrapper: ModuleWrapper, target_name: str, target: Tensor):
    def amin_reduce_func(converted_target: Tensor):
        return converted_target.detach().amin(dim=-1)

    def amax_reduce_func(converted_target: Tensor):
        return converted_target.detach().amax(dim=-1)

    if target_name in wrapper.quantization_target_spaces:
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space._scaler:
            current_amin = target_space._scaler.shrink(target, amin_reduce_func, keepdim=True)
            current_amax = target_space._scaler.shrink(target, amax_reduce_func, keepdim=True)
        else:
            current_amin = target.detach().reshape(-1).amin(-1)
            current_amax = target.detach().reshape(-1).amax(-1)
        target_space.tracked_max = current_amax if target_space.tracked_max is None \
            else update_ema(target_space.tracked_max, current_amax, 0.99)
        target_space.tracked_min = current_amin if target_space.tracked_min is None \
            else update_ema(target_space.tracked_min, current_amin, 0.99)


def update_ema(biased_ema: Tensor, current_val: Tensor, decay: float):
    """
    Exponential moving average method.
    """
    return biased_ema * decay + (1 - decay) * current_val
