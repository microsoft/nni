# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
from typing import List, Dict, Union, overload

import torch
from torch import Tensor

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator
from ..base.target_space import TargetType


_logger = logging.getLogger(__name__)


class DoReFaQuantizer(Quantizer):
    '''
    Dorefa-Quantizer, as defined in:
    `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`__,
    authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.

    Parameters
    ----------
    model
        Model to be quantized.
    config_list
        A list of dict, each dict configure which module need to be quantized, and how to quantize.
        Please refer :doc:`Compression Config Specification </compression/compression_config_list>` for more information.
    evaluator
        TODO: {evaluator_docstring}
    input_layers
        Mark the first layer of the model where the input needs to be quantified.

    Examples
    --------
        >>> from nni.contrib.compression.quantization import DoReFaQuantizer
        >>> from nni.contrib.compression.utils import TorchEvaluator
        >>> model = ...
        >>> optimizer = ...
        >>> max_steps, max_epochs = ..., ...
        >>> evaluator = TorchEvaluator(train, optimizer, training_step)
        >>> quantizer = DoReFaQuantizer(model, configure_list, evaluator)
        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)
    '''
    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, input_layers: List):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 input_layers: List=[], existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 input_layers: List=[], existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False
        self.input_layers = input_layers if input_layers is not None else []

        self.check_validation()
        self.register_dorefa_apply_method()
        self.register_track_func()

    def check_validation(self):
        for module_name, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                is_input_check = module_name in self.input_layers and target_space.type is TargetType.INPUT
                if target_space.quant_scheme != 'affine' and not is_input_check:
                    warn_msg = f"Only supports affine mode for middle layers, bug got {target_space.quant_scheme}"
                    _logger.warning(warn_msg)

    def register_dorefa_apply_method(self):
        for module_name, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.type is TargetType.PARAMETER:
                    target_space.apply_method = 'dorefa_clamp_round_weight'
                elif target_space.type is TargetType.INPUT:
                    if module_name in self.input_layers:
                        target_space.apply_method = 'clamp_round'
                    else:
                        target_space.apply_method = "dorefa_clamp_round_input"
                elif target_space.type is TargetType.OUTPUT:
                    target_space.apply_method = "dorefa_clamp_round_output"

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.initialize_scale_zp)
            wrapper.register_track_func(self.update_scale_zp)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if target_name not in wrapper.quantization_target_spaces or \
            wrapper.name not in self.input_layers:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is not TargetType.INPUT:
            return
        # track min max values
        current_amax, current_amin = track_max_min(wrapper, target_name, target)
        # update scale and zero_point
        tracked_min = torch.min(current_amin, torch.zeros_like(current_amin))
        tracked_max = torch.max(current_amax, torch.zeros_like(current_amax))
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

    def initialize_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init or target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if (target_space.type is TargetType.INPUT and wrapper.name \
            not in self.input_layers) or target_space.type is TargetType.PARAMETER: #zero_point and scale don't change anymore
            tracked_max = torch.tensor(1.0).to(target.device)
            tracked_min = torch.tensor(0.0).to(target.device)
            scale, zero_point = init_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')
        elif target_space.type is TargetType.INPUT and wrapper.name in self.input_layers:
            return
        elif target_space.type is TargetType.OUTPUT:
            tracked_max = torch.tensor(1.0 + 0.5 / (2**target_space.quant_bits - 1)).to(target.device)
            tracked_min = torch.tensor(0 - 0.5 / (2**target_space.quant_bits - 1)).to(target.device)
            scale, zero_point = init_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')
        else:
            raise RuntimeError(f'Unknown target_name {target_name}')

        target_space.scale, target_space.zero_point = scale, zero_point

    def register_trigger(self, evaluator: Evaluator):
        def optimizer_task():
            self.is_init = True

        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        pass


def init_scale_zp(tracked_max: Tensor, tracked_min: Tensor, qmax: int, qmin: int, quant_scheme: Union[str, None] = None):
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


def track_max_min(wrapper: ModuleWrapper, target_name: str, target: Tensor):
    def amin_reduce_func(converted_target: Tensor):
        return converted_target.detach().amin(dim=-1)

    def amax_reduce_func(converted_target: Tensor):
        return converted_target.detach().amax(dim=-1)

    target_space = wrapper.quantization_target_spaces[target_name]
    if target_space._scaler:
        current_amin = target_space._scaler.shrink(target, amin_reduce_func, keepdim=True)
        current_amax = target_space._scaler.shrink(target, amax_reduce_func, keepdim=True)
    else:
        current_amin = target.detach().reshape(-1).amin(-1)
        current_amax = target.detach().reshape(-1).amax(-1)

    return current_amax, current_amin
