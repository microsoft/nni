# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch

from .target_space import PruningTargetSpace, QuantizationTargetSpace, TargetSpace, TargetType


def bypass(target: torch.Tensor, target_space: TargetSpace):
    return target * 1.


def lsq_clamp_round(target: torch.Tensor, target_space: QuantizationTargetSpace):
    qmax: int = target_space.qmax
    qmin: int = target_space.qmin
    if target_space._scaler is not None:
        scale = target_space._scaler.expand(target_space.scale, target_space.shape, keepdim=True) # type: ignore
    else:
        scale = target_space.scale

    class LSQClampRound(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, target: torch.Tensor, scale: torch.Tensor):
            ctx.save_for_backward(target, scale)
            quantize_target = torch.clamp(target / scale, qmin, qmax)
            dequantize_target = torch.round(quantize_target) * scale
            return dequantize_target

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            target, scale = ctx.saved_tensors
            grad_scale_factor = 1.0 / ((qmax * target.numel()) ** 0.5) if (qmax * target.numel()) ** 0.5 != 0 else 1.0

            q_target = target / scale
            # compute index
            ind_neg = (q_target < qmin).float()
            ind_pos = (q_target > qmax).float()
            ind_mid = (1.0 - ind_neg.float() - ind_pos.float())
            # scale gradient
            grad_scale = (ind_neg * qmin + ind_pos * qmax + ind_mid * (-q_target + torch.round(q_target))) * grad_output * grad_scale_factor
            if target_space._scaler is None:
                grad_scale = grad_scale.sum().expand(scale.size())
            grad_target = grad_output * ind_mid

            return grad_target, grad_scale

    return LSQClampRound.apply(target, scale)


def lsq_plus_clamp_round(target: torch.Tensor, target_space: QuantizationTargetSpace):
    qmax: int = target_space.qmax
    qmin: int = target_space.qmin
    if target_space._scaler is not None:
        scale = target_space._scaler.expand(target_space.scale, target_space.shape, keepdim=True) # type: ignore
        zero_point = target_space._scaler.expand(target_space.zero_point, target_space.shape, keepdim=True) # type: ignore
    else:
        scale = target_space.scale
        zero_point = target_space.zero_point

    class LSQPlusClampRound(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, target: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
            ctx.save_for_backward(target, scale, zero_point)
            new_target = torch.clamp(target / scale + zero_point, qmin, qmax)
            dequantized_target = (torch.round(new_target) - zero_point) * scale

            return dequantized_target

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            target, scale, zero_point = ctx.saved_tensors
            grad_scale_factor = 1.0 / ((qmax * target.numel()) ** 0.5) if (qmax * target.numel()) ** 0.5 != 0 else 1.0
            #
            q_target = target / scale
            # compute index
            ind_neg = ((q_target + zero_point) < qmin).float()
            ind_pos = ((q_target + zero_point) > qmax).float()
            ind_mid = (1.0 - ind_neg.float() - ind_pos.float())
            # scale gradient
            grad_scale = (ind_neg * (qmin - zero_point) + ind_pos * (qmax - zero_point) + \
                          ind_mid * (-q_target - zero_point + \
                                     torch.round(q_target + zero_point))) * grad_output * grad_scale_factor
            # zero_point gradient
            grad_zp = (ind_neg * -scale + ind_pos * -scale + ind_mid * 0.0) * grad_output
            # target gradient
            grad_target = grad_output * ind_mid

            if target_space._scaler is None:
                grad_scale = grad_scale.sum().expand(scale.size())
                grad_zp = grad_zp.sum().expand(zero_point.size())

            return grad_target, grad_scale, grad_zp

    return LSQPlusClampRound.apply(target, scale, zero_point)


class DoferaGradClampRound(torch.autograd.Function):
    @staticmethod
    def dorefa_clamp_round_weight(target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        # TODO process special case: quant_bit == 1
        target = target.tanh()
        target = target / (2 * target.abs().max()) + 0.5
        dequantized_target = ClampRound.apply(target, target_space)

        return 2 * dequantized_target - 1  # type: ignore

    @staticmethod
    def dorefa_clamp_round_output(target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
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
        if target_space.type == TargetType.OUTPUT:
            grad_output[torch.abs(target) > 1] = 0

        return grad_output, None


class ClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        if target_space._scaler is not None:
            zero_point = target_space._scaler.expand(target_space.zero_point, target_space.shape, keepdim=True) # type: ignore
            scale = target_space._scaler.expand(target_space.scale, target_space.shape, keepdim=True) # type: ignore
        else:
            zero_point = target_space.zero_point
            scale = target_space.scale
        transformed_target = zero_point + target / scale
        quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
        dequantized_target = (quantized_target - zero_point) * scale
        return dequantized_target

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        return grad_output, None


class QATClampRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: torch.Tensor, target_space: QuantizationTargetSpace) -> Any:
        if target_space._scaler is not None:
            zero_point = target_space._scaler.expand(target_space.zero_point, target_space.shape, keepdim=True) # type: ignore
            scale = target_space._scaler.expand(target_space.scale, target_space.shape, keepdim=True) # type: ignore
        else:
            zero_point = target_space.zero_point
            scale = target_space.scale

        transformed_target = zero_point + target / scale
        quantized_target = torch.round(torch.clamp(transformed_target, target_space.qmin, target_space.qmax))
        dequantized_target = (quantized_target - zero_point) * scale
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


class _StraightThrough(torch.autograd.Function):
    """
    Straight through the gradient to the score, then the score = initial_score + sum(-lr * grad(weight) * weight).
    """
    @staticmethod
    def forward(ctx, score, mask):
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


def movement_mul_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    score = getattr(target_space._wrapper, f'{target_space._target_name}_mvp_score', None)
    if score is None:
        return mul_mask(target, target_space)
    else:
        assert target_space.mask is not None and target_space.shape is not None
        if target_space._scaler is not None:
            score = target_space._scaler.expand(score, target_space.shape, keepdim=True, full_expand=False)
        return torch.mul(target, _StraightThrough.apply(score, target_space.mask))  # type: ignore


def movement_add_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    score = getattr(target_space._wrapper, f'{target_space._target_name}_mvp_score', None)
    if score is None:
        return add_mask(target, target_space)
    else:
        assert target_space.mask is not None and target_space.shape is not None
        trans_mask = torch.where(target_space.mask == 1, torch.zeros_like(target_space.mask), SMALL_MASK_VALUE)
        if target_space._scaler is not None:
            score = target_space._scaler.expand(score, target_space.shape, keepdim=True, full_expand=False)
        return torch.add(target, _StraightThrough.apply(score, trans_mask))  # type: ignore


def slim_mul_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    scaling_factor = getattr(target_space._wrapper, f'{target_space._target_name}_slim_factor', None)
    if scaling_factor is None:
        return mul_mask(target, target_space)
    else:
        assert target_space.shape is not None
        if target_space._scaler is not None:
            scaling_factor = target_space._scaler.expand(scaling_factor, target_space.shape, keepdim=True, full_expand=False)
        return mul_mask(torch.mul(target, scaling_factor), target_space)


pruning_apply_methods = {
    'bypass': bypass,
    'mul': mul_mask,
    'add': add_mask,
    'movement_mul': movement_mul_mask,
    'movement_add': movement_add_mask,
    'slim_mul': slim_mul_mask
}


quant_apply_methods = {
    'bypass': bypass,
    'clamp_round': ClampRound.apply,
    'qat_clamp_round': QATClampRound.apply,
    'dorefa_clamp_round_weight': DoferaGradClampRound.dorefa_clamp_round_weight,
    'dorefa_clamp_round_output': DoferaGradClampRound.dorefa_clamp_round_output,
    "lsq_clamp_round": lsq_clamp_round,
    'bnn_clamp_round': BNNClampRound.apply,
    'lsq_plus_clamp_round': lsq_plus_clamp_round
}
