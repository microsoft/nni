# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch

from .target_space import PruningTargetSpace, QuantizationTargetSpace, TargetSpace


def bypass(target: torch.Tensor, target_space: TargetSpace):
    return target * 1.


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
            score = target_space._scaler.expand(score, target_space.shape)
        return torch.mul(target, _StraightThrough.apply(score, target_space.mask))


def movement_add_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    score = getattr(target_space._wrapper, f'{target_space._target_name}_mvp_score', None)
    if score is None:
        return add_mask(target, target_space)
    else:
        assert target_space.mask is not None and target_space.shape is not None
        trans_mask = torch.where(target_space.mask == 1, torch.zeros_like(target_space.mask), SMALL_MASK_VALUE)
        if target_space._scaler is not None:
            score = target_space._scaler.expand(score, target_space.shape)
        return torch.add(target, _StraightThrough.apply(score, trans_mask))


def slim_mul_mask(target: torch.Tensor, target_space: PruningTargetSpace):
    scaling_factor = getattr(target_space._wrapper, f'{target_space._target_name}_slim_factor', None)
    if scaling_factor is None:
        return mul_mask(target, target_space)
    else:
        assert target_space.shape is not None
        if target_space._scaler is not None:
            scaling_factor = target_space._scaler.expand(scaling_factor, target_space.shape)
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
}
