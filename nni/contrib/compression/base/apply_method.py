# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch

from .target_space import QuantizationTargetSpace


class ClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target, target_space: QuantizationTargetSpace) -> Any:
        transformed_target = target_space.zero_point + target / target_space.scale
        quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
        dequantized_target = (quantized_target - target_space.zero_point) * target_space.scale
        return dequantized_target

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        return grad_output, None


class QATClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target, target_space: QuantizationTargetSpace) -> Any:
        transformed_target = target_space.zero_point + target / target_space.scale
        quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
        dequantized_target = (quantized_target - target_space.zero_point) * target_space.scale
        ctx.save_for_backward(transformed_target)
        ctx.target_space = target_space
        return dequantized_target

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        transformed_target, = ctx.saved_variables
        target_space = ctx.target_space
        mask = (transformed_target < target_space.qmin) | (transformed_target > target_space.qmax)
        grad_output[mask] = 0.
        return grad_output, None


quant_apply_methods = {
    'clamp_round': ClampRound.apply,
    'qat_clamp_round': QATClampRound.apply,
}
