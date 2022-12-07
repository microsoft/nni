# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from collections import defaultdict
import heapq
from typing import Callable, Dict, List, Tuple

import torch

from ..base.target_space import PruningTargetSpace


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

    dependency_target_spaces, remained_target_spaces = target_spaces_filter(target_spaces, condition_dependency)

    global_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_global)
    ratio_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_ratio)
    threshold_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_threshold)
    align_target_spaces, remained_target_spaces = target_spaces_filter(remained_target_spaces, condition_align)


def target_spaces_filter(target_spaces: _TARGET_SPACES, condition: Callable[[PruningTargetSpace], bool]) -> _TARGET_SPACES:
    filtered_target_spaces = defaultdict(dict)
    remained_target_spaces = defaultdict(dict)

    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            if condition(target_space):
                filtered_target_spaces[module_name][target_name] = target_space
            else:
                remained_target_spaces[module_name][target_name] = target_space
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
    pass


def _generate_global_sparsity(metrics: _METRICS, target_spaces: _TARGET_SPACES) -> _MASKS:
    groups = defaultdict(list)
    for module_name, ts in target_spaces.items():
        for target_name, target_space in ts.items():
            groups[module_name][target_name] = target_space

    masks = defaultdict(dict)
    for _, group in groups.items():
        if len(group) == 0:
            continue

        group_sparse_ratio = group[0][2].sparse_ratio
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


def __generate_dependency_sparsity(metrics: _METRICS, groups: Dict[str, List[Tuple[str, str, PruningTargetSpace]]]) -> _MASKS:
    pass


# the following are helper functions

def _ratio_mask(metric: torch.Tensor, sparse_ratio: float):
    if sparse_ratio == 0.0:
        return torch.ones_like(metric)

    if sparse_ratio == 1.0:
        return torch.zeros_like(metric)

    assert 0.0 < sparse_ratio < 1.0
    sparse_number = int(sparse_ratio * metric.numel())
    _, indices = metric.topk(sparse_number, largest=False)
    return torch.ones_like(metric).scatter(-1, indices, 1.0)


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


def _nested_update(default_dict: _MASKS, update_dict: _MASKS):
    for key, value in update_dict.items():
        for k, v in value.items():
            default_dict[key][k] = v
