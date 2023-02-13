# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch

from .setting import OUTPUT_PREFIX
from .target_space import PruningTargetSpace, QuantizationTargetSpace


def bypass(target: torch.Tensor, target_space: QuantizationTargetSpace):
    return target * 1.


def lsq_clamp_round(target: torch.Tensor, target_space: QuantizationTargetSpace):
    def grad_scale(x, scale_factor):
        y_out = x
        y_grad = x * scale_factor
        return (y_out - y_grad).detach() + y_grad

    def round_pass(x):
        y_out = torch.round(x)
        y_grad = x
        return (y_out - y_grad).detach() + y_grad

    qmax, qmin = target_space.qmax, target_space.qmin
    #Quantize
    grad_scale_factor = 1.0 / ((qmax * target.numel()) ** 0.5) if (qmax * target.numel()) ** 0.5 != 0 else 1.0
    scale = grad_scale(target_space.scale, grad_scale_factor)
    new_target = torch.clamp(target / scale, qmin, qmax)
    dequantized_target = round_pass(new_target) * scale

    return dequantized_target

class DoferaGradClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        ctx.target_space = target_space
        ctx.save_for_backward(target)
        return target * 1

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        target_space = ctx.target_space
        target, = ctx.saved_variables
        grad_o = torch.abs(grad_output.detach())
        dim_lis = list(range(len(grad_o.shape)))
        dim_lis.pop(0)
        max_grad = torch.amax(grad_o, dim=dim_lis, keepdim=True)
        # generate uniform noise
        uniform_k = torch.zeros_like(max_grad).to(target.device)
        N_k = uniform_k.uniform_(-0.5, 0.5) / (2**(target_space.quant_bits) - 1)
        q_grad_o = grad_output / (2 * max_grad) + 0.5 + N_k
        quantized_grad = target_space.zero_point + q_grad_o / target_space.scale
        quantized_grad = torch.round(torch.clamp(quantized_grad, target_space.qmin, target_space.qmax))
        dequantized_grad = (quantized_grad - target_space.zero_point) * target_space.scale

        return (dequantized_grad - 0.5) * 2 * max_grad, None

    @staticmethod
    def dorefa_clamp_round_weight(target: torch.Tensor, target_space: QuantizationTargetSpace):
        # TODO process special case: quant_bit == 1
        target = target.tanh()
        target = target / (2 * target.abs().max()) + 0.5
        dequantized_target = ClampRound.apply(target, target_space)

        return 2 * dequantized_target - 1

    @staticmethod
    def dorefa_clamp_round_input(target: torch.Tensor, target_space: QuantizationTargetSpace):
        target = torch.clamp(target, 0, 1)
        return ClampRound.apply(target, target_space)


class BNNClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        ctx.save_for_backward(target)
        signed_target = torch.sign(target)
        signed_target[signed_target == 0] = 1
        ctx.target_space = target_space
        return signed_target

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        target, = ctx.saved_variables
        target_space = ctx.target_space
        if OUTPUT_PREFIX in target_space._target_name:
            grad_output[torch.abs(target) > 1] = 0

        return grad_output, None


class ClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        transformed_target = target_space.zero_point + target / target_space.scale
        quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
        dequantized_target = (quantized_target - target_space.zero_point) * target_space.scale
        return dequantized_target

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        return grad_output, None


class QATClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
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


SMALL_MASK_VALUE = -1000.0


def mul_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    assert target_space.mask is not None
    return torch.mul(target, target_space.mask)


def add_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    assert target_space.mask is not None
    trans_mask = torch.where(target_space.mask == 1, torch.zeros_like(target_space.mask), SMALL_MASK_VALUE)
    return torch.add(target, trans_mask)


pruning_apply_methods = {
    'bypass': bypass,
    'mul': mul_mask,
    'add': add_mask,
}


quant_apply_methods = {
    'bypass': bypass,
    'clamp_round': ClampRound.apply,
    'qat_clamp_round': QATClampRound.apply,
    'dofera_clamp_round_weight': DoferaGradClampRound.dorefa_clamp_round_weight,
    'dofera_clamp_round_input': DoferaGradClampRound.dorefa_clamp_round_input,
    'dofera_clamp_round_output': DoferaGradClampRound.apply,
    "lsq_clamp_round": lsq_clamp_round,
    'bnn_clamp_round': BNNClampRound.apply,
}
