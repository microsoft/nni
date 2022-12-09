# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import heapq
from typing import Callable, Dict, List, Tuple

import numpy
import torch

from ..base.target_space import PruningTargetSpace, TargetType


_TARGET_SPACES = Dict[str, Dict[str, PruningTargetSpace]]
_MASKS = Dict[str, Dict[str, torch.Tensor]]
_METRICS = Dict[str, Dict[str, torch.Tensor]]


def active_sparse_targets_filter(target_spaces: _TARGET_SPACES) -> _METRICS:
    # filter all targets need to active generate sparsity
    active_targets = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if target_space.sparse_ratio or target_space.sparse_threshold:
                active_targets[module_name][target_name] = target_space.target
    return active_targets


def generate_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    def condition_dependency(target_space: PruningTargetSpace) -> bool:
        return target_space.dependency_group_id is not None

    def condition_global(target_space: PruningTargetSpace) -> bool:
        return target_space.global_group_id is not None

    def condition_ratio(target_space: PruningTargetSpace) -> bool:
        return target_space.sparse_ratio is not None

    def condition_threshold(target_space: PruningTargetSpace) -> bool:
        return target_space.sparse_threshold is not None

    def condition_align(target_space: PruningTargetSpace) -> bool:
        return target_space.align is not None

    masks = defaultdict(dict)

    threshold_target_spaces, remained_target_spaces = target_spaces_filter(target_spaces, condition_threshold)
    update_masks = _generate_threshold_sparsity(metrics, threshold_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, threshold_target_spaces))

    dependency_target_spaces, remained_target_spaces = target_spaces_filter(target_spaces, condition_dependency)
    update_masks = _generate_dependency_sparsity(metrics, dependency_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, dependency_target_spaces))

    global_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_global)
    update_masks = _generate_global_sparsity(metrics, global_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, global_target_spaces))

    ratio_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_ratio)
    update_masks = _generate_ratio_sparsity(metrics, ratio_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, ratio_target_spaces))

    align_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_align)
    update_masks = _generate_align_sparsity(masks, align_target_spaces)
    _nested_multiply_update_masks(masks, _expand_masks(update_masks, align_target_spaces))

    return masks


def target_spaces_filter(target_spaces: _TARGET_SPACES,
                         condition: Callable[[PruningTargetSpace], bool]) -> Tuple[_TARGET_SPACES, _TARGET_SPACES]:
    filtered_target_spaces = defaultdict(dict)
    remained_target_spaces = defaultdict(dict)

    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if (target_space.type is TargetType.PARAMETER and target_space.target is None) or not condition(target_space):
                remained_target_spaces[module_name][target_name] = target_space
            else:
                filtered_target_spaces[module_name][target_name] = target_space

    return filtered_target_spaces, remained_target_spaces


def _generate_ratio_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    # NOTE: smaller metric value means more un-important
    masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            metric = metrics[module_name][target_name]
            min_sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
            max_sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 1.0
            sparse_ratio = min(max_sparse_ratio, max(min_sparse_ratio, target_space.sparse_ratio))
            masks[module_name][target_name] = _ratio_mask(metric, sparse_ratio)
    return masks


def _generate_threshold_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    # NOTE: smaller metric value means more un-important
    masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            metric = metrics[module_name][target_name]
            # metric < threshold will be 0, metric >= threshold will be 1
            mask = _threshold_mask(metric, target_space.sparse_threshold)

            # if sparse_ratio does not meet `min_sparse_ratio`, `max_sparse_ratio`, re-generate mask
            sparse_ratio = 1.0 - mask.sum() / mask.numel()
            min_sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
            max_sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 1.0
            if sparse_ratio < min_sparse_ratio:
                mask = _ratio_mask(metric, min_sparse_ratio)
            if sparse_ratio > max_sparse_ratio:
                mask = _ratio_mask(metric, max_sparse_ratio)

            masks[module_name][target_name] = mask
    return masks


def _generate_align_sparsity(masks: _MASKS, target_spaces: _TARGET_SPACES) -> _MASKS:
    align_masks = defaultdict(dict)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            src_mask = masks[module_name][target_space.align['target_name']]
            align_dims: List[int] = target_space.align['dims']
            reduce_dims = [d for d in range(len(src_mask.shape)) if d not in align_dims and d - len(src_mask.shape) not in align_dims]
            align_masks[module_name][target_name] = src_mask.sum(reduce_dims).bool().float()
    return align_masks


def _generate_global_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    groups: Dict[str, List[Tuple[str, str, PruningTargetSpace]]] = defaultdict(list)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            groups[target_space.global_group_id].append((module_name, target_name, target_space))

    masks = defaultdict(dict)
    for _, group in groups.items():
        group_sparse_ratio = None
        for _, _, target_space in group:
            if target_space.sparse_ratio is not None:
                if group_sparsity_ratio is None:
                    group_sparsity_ratio = target_space.sparse_ratio
                else:
                    assert group_sparsity_ratio == target_space.sparse_ratio
        assert group_sparse_ratio is not None

        # at least how many elements to mask
        sparse_number_low = 0
        # at most how many elements to mask
        sparse_number_high = 0
        # how many elements in this group
        total_element_number = 0
        for _, _, target_space in group:
            element_number = target_space.target.numel()
            total_element_number += element_number
            sparse_number_low += int(element_number * target_space.min_sparse_ratio) if target_space.min_sparse_ratio else 0
            sparse_number_high += int(element_number * target_space.max_sparse_ratio) if target_space.max_sparse_ratio else element_number
        # how many elements should be masked, controlled by sparse_ratio
        sparse_number = int(total_element_number * group_sparse_ratio)

        if sparse_number <= sparse_number_low:
            for module_name, target_name, target_space in group:
                sparse_ratio = target_space.min_sparse_ratio if target_space.min_sparse_ratio else 0.0
                masks[module_name][target_name] = _ratio_mask(metrics[module_name][target_name], sparse_ratio)
            continue

        if sparse_number >= sparse_number_high:
            for module_name, target_name, target_space in group:
                sparse_ratio = target_space.max_sparse_ratio if target_space.max_sparse_ratio else 0.0
                masks[module_name][target_name] = _ratio_mask(metrics[module_name][target_name], sparse_ratio)
            continue

        sparse_threshold = _global_threshold_generate(metrics, group, sparse_number)
        for module_name, target_name, target_space in group:
            masks[module_name][target_name] = _threshold_mask(metrics[module_name][target_name], sparse_threshold)
        continue
    return masks


def _generate_dependency_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    groups: Dict[str, List[Tuple[str, str, PruningTargetSpace]]] = defaultdict(list)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            groups[target_space.dependency_group_id].append((module_name, target_name, target_space))

    masks = defaultdict(dict)
    for _, group in groups.items():
        block_numbers = []
        group_sparsity_ratio = None
        filtered_metrics = defaultdict(dict)

        for module_name, target_name, target_space in group:
            assert isinstance(target_space.internal_metric_block, (int, None))
            block_numbers.append(target_space.internal_metric_block if target_space.internal_metric_block else 1)
            if target_space.sparse_ratio is not None:
                if group_sparsity_ratio is None:
                    group_sparsity_ratio = target_space.sparse_ratio
                else:
                    assert group_sparsity_ratio == target_space.sparse_ratio
            filtered_metrics[module_name][target_name] = metrics[module_name][target_name]
        block_number = numpy.lcm(block_numbers)
        assert group_sparsity_ratio is not None
        group_metric = _metric_fuse(filtered_metrics)
        group_mask = _ratio_mask(group_metric, group_sparsity_ratio, view_size=[block_number, -1])

        for module_name, target_name, _ in group:
            masks[module_name][target_name] = group_mask.clone()

    return masks


# the following are helper functions

def _ratio_mask(metric: torch.Tensor, sparse_ratio: float, view_size: int | List[int] = -1):
    if sparse_ratio == 0.0:
        return torch.ones_like(metric)

    if sparse_ratio == 1.0:
        return torch.zeros_like(metric)

    assert 0.0 < sparse_ratio < 1.0
    if isinstance(view_size, int) or len(view_size[:-1]) == 0:
        block_number = 1
    else:
        block_number = numpy.prod(view_size[:-1])
    sparse_number_per_block = int(metric.numel() // block_number * sparse_ratio)
    viewed_metric = metric.view(view_size)
    _, indices = viewed_metric.topk(sparse_number_per_block, largest=False)
    return torch.ones_like(viewed_metric).scatter(-1, indices, 0.0).reshape_as(metric)


def _threshold_mask(metric: torch.Tensor, sparse_threshold: float):
    return (metric >= sparse_threshold).float().to(metric.device)


def _global_threshold_generate(metrics: _METRICS,
                               group: List[Tuple[str, str, PruningTargetSpace]],
                               sparse_number: int) -> float:
    buffer = []
    buffer_elem = 0
    for module_name, target_name, target_space in group:
        metric = metrics[module_name][target_name]
        grain_size = target_space.target.numel() // metric.numel()
        for m in metric.cpu().detach().view(-1):
            if buffer_elem <= sparse_number:
                heapq.heappush(buffer, (-m.item(), grain_size))
                buffer_elem += grain_size
            else:
                _, previous_grain_size = heapq.heappushpop(buffer, (-m.item(), grain_size))
                buffer_elem += grain_size - previous_grain_size
    return -heapq.heappop(buffer)[0]


def _nested_multiply_update_masks(default_dict: _MASKS, update_dict: _MASKS):
    # if a target already has a mask, the old one will multiply the new one as the target mask,
    # that means the mask in default dict will more and more sparse.
    for key, value in update_dict.items():
        for k, v in value.items():
            if k in default_dict[key] and isinstance(default_dict[key][k], torch.Tensor):
                default_dict[key][k] = (default_dict[key][k] * v).bool().float()
            else:
                default_dict[key][k] = v


def _metric_fuse(metrics: _METRICS) -> torch.Tensor:
    # mean all metric value
    fused_metric = None
    count = 0
    for _, module_metrics in metrics.items():
        for _, target_metric in module_metrics.items():
            if fused_metric:
                fused_metric += target_metric
            else:
                fused_metric = target_metric.clone()
            count += 1
    return fused_metric / count


def _expand_masks(masks: _MASKS, target_spaces: _TARGET_SPACES) -> _MASKS:
    new_masks = defaultdict(dict)
    for module_name, module_masks in masks.items():
        for target_name, target_mask in module_masks.items():
            target_space = target_spaces[module_name][target_name]
            if target_space._scaler:
                new_masks[module_name][target_name] = target_space._scaler.expand(target_mask, target_space.target.shape)
            else:
                new_masks[module_name][target_name] = target_mask
    return new_masks
