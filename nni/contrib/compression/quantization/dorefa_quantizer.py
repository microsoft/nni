# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations
import logging
from typing import List, Dict, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

from nni.common.version import check_torch_version

from ..base.compressor import Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils.evaluator import Evaluator
from ..base.target_space import TargetType, QuantizationTargetSpace

ACTIVATION_LIST = [
    nn.ReLU, nn.RReLU, nn.LeakyReLU, nn.PReLU, nn.Softplus, nn.ELU, nn.CELU, nn.SELU, nn.GELU, \
    nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.Softsign, nn.Hardtanh, nn.Threshold, nn.Tanhshrink, \
    nn.Softshrink, nn.Hardshrink, nn.LogSigmoid, nn.Softmin, nn.Softmax, nn.LogSoftmax, nn.Hardswish,
]


_logger = logging.getLogger(__name__)
is_proper_torch_version = check_torch_version()


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
                assert target_space.quant_scheme != None
                if target_space.type is TargetType.PARAMETER and target_space.quant_scheme != 'affine':
                    warn_msg = f"Only supports affine mode for weight quantization, bug got {target_space.quant_scheme}"
                    _logger.warning(warn_msg)
                elif target_space.type is TargetType.OUTPUT:
                    module = target_space._wrapper.module
                    # case 1: activation module
                    # case 2: module with activation fused_modules
                    fused_modules = target_space._wrapper.fused_modules if target_space._wrapper.fused_modules else []
                    if not isinstance(module, tuple(ACTIVATION_LIST)) and not (len(fused_modules) > 0 and # type: ignore
                        any([isinstance(item, tuple(ACTIVATION_LIST)) for item in fused_modules[1:]])): # type: ignore
                        raise ValueError("Output quantization is only supported for activation function or" + \
                                          f"activation module fusion, but got {type(module)}")
                    if target_space.quant_scheme != 'affine':
                        warn_msg = f"Only supports affine mode for output quantization, bug got {target_space.quant_scheme}"
                        _logger.warning(warn_msg)
                if target_space._scaler is not None:
                    raise ValueError("DoRefa Qauntizer doesn't support for granularity, please set it to False")

    def _quant_dequant_gradient_hook(self, target_space: QuantizationTargetSpace):
        def quant_dequant_gradient(module: nn.Module, grad_output):
            tracked_max = torch.tensor(1.0 + 0.5 / (2**target_space.quant_bits - 1)).to(grad_output[0].device)
            tracked_min = torch.tensor(0 - 0.5 / (2**target_space.quant_bits - 1)).to(grad_output[0].device)
            scale, zero_point = init_scale_zp(tracked_max, tracked_min, target_space.qmax, \
                                target_space.qmin, 'affine')
            new_grad_output = []
            for g_o in grad_output:
                grad_o = torch.abs(g_o.clone().detach())
                dim_lis = list(range(len(grad_o.shape)))
                dim_lis.pop(0)
                max_grad = torch.amax(grad_o, dim=dim_lis, keepdim=True)
                # generate uniform noise
                uniform_k = torch.zeros_like(max_grad).to(g_o.device)
                N_k = uniform_k.uniform_(-0.5, 0.5) / (2**(target_space.quant_bits) - 1)
                q_grad_o = g_o / (2 * max_grad) + 0.5 + N_k
                quantized_grad = zero_point + q_grad_o / scale
                quantized_grad = torch.round(torch.clamp(quantized_grad, target_space.qmin, target_space.qmax))
                dequantized_grad = (quantized_grad - zero_point) * scale
                new_grad_output.append((dequantized_grad - 0.5) * 2 * max_grad)

            return tuple(new_grad_output)

        target_space._wrapper.module.register_full_backward_pre_hook(quant_dequant_gradient) # type: ignore

    def register_output_backward_hook(self):
        for _, ts in self._target_spaces.items():
            is_output = any([target_space.type is TargetType.OUTPUT for target_space in ts.values()])
            is_param = any([target_space.type is TargetType.PARAMETER for target_space in ts.values()])
            if is_param and not is_output:
                if is_proper_torch_version: # torch version >= 2.0.0
                    for _, target_space in ts.items():
                        if target_space.type is TargetType.PARAMETER:
                            self._quant_dequant_gradient_hook(target_space)
                            break
                else:
                    warn_msg = f"Gradient quantization is only supported for torch version >= 2.0.0"
                    _logger.warning(warn_msg)

    def register_dorefa_apply_method(self):
        for _, ts in self._target_spaces.items():
            for _, target_space in ts.items():
                if target_space.type is TargetType.PARAMETER:
                    target_space.apply_method = 'dorefa_clamp_round_weight'
                elif target_space.type is TargetType.INPUT:
                    target_space.apply_method = 'clamp_round'
                elif target_space.type is TargetType.OUTPUT:
                    target_space.apply_method = "dorefa_clamp_round_output"

    def register_track_func(self):
        for module_name, _ in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.initialize_scale_zp)
            wrapper.register_track_func(self.update_scale_zp)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is not TargetType.INPUT:
            return
        # track min max values
        current_amin = target.detach().reshape(-1).amin(-1)
        current_amax = target.detach().reshape(-1).amax(-1)
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
        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is TargetType.INPUT:
            return
        elif target_space.type in [TargetType.OUTPUT, TargetType.PARAMETER]:
            tracked_max = torch.tensor(1.0).to(target.device)
            tracked_min = torch.tensor(0.0).to(target.device)
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
        self.register_output_backward_hook()
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
