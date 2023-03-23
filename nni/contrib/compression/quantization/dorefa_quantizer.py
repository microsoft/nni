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
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator,
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, \
                 existed_wrappers: Dict[str, ModuleWrapper] | None = None):
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False

        self.check_validation()
        self.register_dorefa_apply_method()
        self.register_track_func()

    def check_validation(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.quant_scheme != 'affine':
                    warn_msg = f"Only supports affine mode, bug got {target_space.quant_scheme}"
                    _logger.warning(warn_msg)

    def register_dorefa_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.type is TargetType.PARAMETER:
                    target_space.apply_method = 'dofera_clamp_round_weight'
                elif target_space.type is TargetType.INPUT:
                    target_space.apply_method = "dofera_clamp_round_input"
                elif target_space.type is TargetType.OUTPUT:
                    target_space.apply_method = "dofera_clamp_round_output"

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.initialize_scale_zp)

    def initialize_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if self.is_init or target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is TargetType.INPUT or "weight" in target_name: #zero_point and scale don't change anymore
            tracked_max = torch.tensor(1.0).to(target.device)
            tracked_min = torch.tensor(0.0).to(target.device)
            scale, zero_point = init_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')
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
