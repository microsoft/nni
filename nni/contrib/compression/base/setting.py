# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Literal, Type

import torch

INPUT_PREFIX = '_input_'
OUTPUT_PREFIX = '_output_'

_SETTING = Dict[str, Dict[str, Any]]

class ModuleSetting:
    """
    This class is used to get the compression setting for a module.
    The setting configures the relevant basic information of how to compress the current module.
    """

    # {module_cls_name: {target_name: target_setting}}
    registry: Dict[str, _SETTING]

    @classmethod
    def register(cls, module_cls__name: str, setting: _SETTING):
        cls.registry[module_cls__name] = deepcopy(setting)

    @classmethod
    def get(cls, module_cls_or_name: str | torch.nn.Module | Type[torch.nn.Module], target_names: List[str] | None = None,
            update_setting: Dict[str, Any] | None = None) -> _SETTING:
        """
        Return the updated setting for the given module type.

        Parameters
        ----------
        module
            A module instance or a module class.
        target_names
            The name of the targets to be compressed.
        update_setting
            Used to update the given module type setting from the registry.
        """
        if isinstance(module_cls_or_name, torch.nn.Module):
            module_cls_name = module_cls_or_name.__class__.__name__
        elif issubclass(module_cls_or_name, torch.nn.Module):
            module_cls_name = module_cls_or_name.__name__
        elif isinstance(module_cls_or_name, str):
            module_cls_name = module_cls_or_name
        else:
            raise RuntimeError('`module_cls_or_name` is not an instance or subclass of `torch.nn.Module` or a name.')

        target_settings = deepcopy(cls.registry.get(module_cls_name, None))
        assert target_settings is not None, \
            f'{module_cls_name} is not registered, please register setting with {cls.__name__}.register().'

        cls._update_setting(target_settings, update_setting)

        target_names = list(target_names if target_names else target_settings.keys())

        # by default, if INPUT_PREFIX or OUTPUT_PREFIX is set in target_names, assuem it points to the first var of the input or output.
        if INPUT_PREFIX in target_names:
            target_names.remove(INPUT_PREFIX)
            target_names.append(f'{INPUT_PREFIX}0')
        if OUTPUT_PREFIX in target_names:
            target_names.remove(OUTPUT_PREFIX)
            target_names.append(f'{OUTPUT_PREFIX}0')

        selected_settings = {}
        for target_name in target_names:
            if target_name.startswith(INPUT_PREFIX):
                selected_settings[target_name] = deepcopy(target_settings.get(target_name, target_settings[INPUT_PREFIX]))
            elif target_name.startswith(OUTPUT_PREFIX):
                selected_settings[target_name] = deepcopy(target_settings.get(target_name, target_settings[OUTPUT_PREFIX]))
            else:
                selected_settings[target_name] = deepcopy(target_settings[target_name])

        return selected_settings

    @classmethod
    def _update_setting(cls, setting1: _SETTING, setting2: _SETTING):
        for target_name, target_setting in setting2.items():
            if target_name in setting1:
                setting1[target_name].update(deepcopy(target_setting))
            else:
                setting1[target_name] = deepcopy(target_setting)


class PruningSetting(ModuleSetting):
    default_setting = {
        'weight': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'sparse_granularity': None,
            'apply_method': 'mul'
        },
        'bias': {
            'align': {
                'target_name': 'weight',
                'dim': 0
            },
            'apply_method': 'mul'
        }
    }

    registry = {
        'Conv2d': default_setting,
        'Linear': default_setting
    }


class QuantizationSetting(ModuleSetting):
    default_setting = {
        '_input_': {
            'quant_dtype': None,
            'apply_method': 'clamp_round'
        },
        'weight': {
            'quant_dtype': None,
            'apply_method': 'clamp_round'
        },
        'bias': {
            'quant_dtype': None,
            'apply_method': 'clamp_round'
        },
        '_output_': {
            'quant_dtype': None,
            'apply_method': 'clamp_round'
        }
    }

    registry = {
        'Conv2d': default_setting,
        'Linear': default_setting
    }


class DistillatoinSetting(ModuleSetting):
    pass


def canonicalize_settings(module: torch.nn.Module,
                          config: Dict[str, Any],
                          mode: Literal['pruning', 'quantization', 'distillation'] | None = None) -> Dict[str, Dict[str, Any]]:
    assert mode is not None
    if mode == 'pruning':
        return PruningSetting.get(module, config.get('target_names', None), config.get('target_settings', None))
    if mode == 'quantization':
        return QuantizationSetting.get(module, config.get('target_names', None), config.get('target_settings', None))
    if mode == 'distillation':
        return DistillatoinSetting.get(module, config.get('target_names', None), config.get('target_settings', None))
