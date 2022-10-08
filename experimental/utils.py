from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, Literal, Tuple

import torch

from .settings import PruningSetting, QuantizationSetting, DistillatoinSetting


def select_modules(model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Any]]:
    # return ({module_name: module}, public_config)
    # intersection(union(op_names, op_names_re), op_types) - exclude_op_names - exclude_op_names_re - exclude_op_types
    name2module = {}
    type2names = defaultdict(set)
    for module_name, module in model.named_modules(remove_duplicate=False):
        name2module[module_name] = module
        type2names[type(module).__name__].add(module_name)

    config = deepcopy(config)
    op_names = set(config.pop('op_names', list()))
    op_types = config.pop('op_types', list())
    op_names_re = config.pop('op_names_re', list())
    exclude_op_names = config.pop('exclude_op_names', list())
    exclude_op_types = config.pop('exclude_op_types', list())
    exclude_op_names_re = config.pop('exclude_op_names_re', list())

    for op_name_re in op_names_re:
        for op_name in name2module:
            if re.match(op_name_re, op_name):
                op_names.add(op_name)

    for op_type in op_types:
        op_names.intersection_update(type2names.get(op_type, set()))

    op_names.difference_update(exclude_op_names)

    for op_name_re in exclude_op_names_re:
        for op_name in name2module:
            if re.match(op_name_re, op_name) and op_name in op_names:
                op_names.remove(op_name)

    for op_type in exclude_op_types:
        op_names.difference_update(type2names.get(op_type, set()))

    return {module_name: name2module[module_name] for module_name in op_names}, config


def canonicalize_settings(module: torch.nn.Module,
                          config: Dict[str, Any],
                          mode: Literal['pruning', 'quantization', 'distillation'] | None = None) -> Dict[str, Dict[str, Any]]:
    assert mode is not None
    if mode == 'pruning':
        return PruningSetting.get(module, config)
    if mode == 'quantization':
        return QuantizationSetting.get(module, config)
    if mode == 'distillation':
        return DistillatoinSetting.get(module, config)
