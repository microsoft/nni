# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Type

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
    def get(cls, module_cls_or_name: str | torch.nn.Module | Type[torch.nn.Module], config: Dict[str, Any]) -> _SETTING:
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
        module_cls_name = cls._get_module_cls_name(module_cls_or_name)
        # deepcopy because config might be pop key or modified context
        config = deepcopy(config)

        target_names = config.pop('target_names', None)
        update_settings = config.pop('target_settings', None)
        target_settings = deepcopy(cls.registry.get(module_cls_name, None))
        assert target_settings is not None, \
            f'{module_cls_name} is not registered, please register setting with {cls.__name__}.register().'

        cls._update_shortcut_setting(target_settings, config)

        if update_settings:
            cls._update_setting(target_settings, update_settings)

        target_names = list(target_names if target_names else target_settings.keys())

        # by default, if INPUT_PREFIX or OUTPUT_PREFIX is set in target_names,
        # assuem it points to the first parameter of the input or output.
        if INPUT_PREFIX in target_names:
            target_names.remove(INPUT_PREFIX)
            target_names.append(f'{INPUT_PREFIX}0')
        if OUTPUT_PREFIX in target_names:
            target_names.remove(OUTPUT_PREFIX)
            target_names.append(f'{OUTPUT_PREFIX}0')

        selected_settings = {}
        for target_name in target_names:
            # TODO: should target_settings[xx_PREFIX] merge with target_settings[f'{xx_PREFIX}{idx_or_name}']?
            if target_name.startswith(INPUT_PREFIX):
                selected_settings[target_name] = deepcopy(target_settings.get(target_name, target_settings[INPUT_PREFIX]))
            elif target_name.startswith(OUTPUT_PREFIX):
                selected_settings[target_name] = deepcopy(target_settings.get(target_name, target_settings[OUTPUT_PREFIX]))
            else:
                selected_settings[target_name] = deepcopy(target_settings[target_name])

        return selected_settings

    @classmethod
    def _get_module_cls_name(cls, module_cls_or_name: str | torch.nn.Module | Type[torch.nn.Module]) -> str:
        if isinstance(module_cls_or_name, torch.nn.Module):
            module_cls_name = module_cls_or_name.__class__.__name__
        elif isinstance(module_cls_or_name, str):
            module_cls_name = module_cls_or_name
        elif issubclass(module_cls_or_name, torch.nn.Module):
            module_cls_name = module_cls_or_name.__name__
        else:
            raise RuntimeError('`module_cls_or_name` is not an instance or subclass of `torch.nn.Module` or a name.')
        return module_cls_name

    @classmethod
    def _update_setting(cls, setting1: _SETTING, setting2: _SETTING):
        for target_name, target_setting in setting2.items():
            if target_name in setting1:
                setting1[target_name].update(deepcopy(target_setting))
            else:
                setting1[target_name] = deepcopy(target_setting)

    @classmethod
    def _update_shortcut_setting(cls, setting1: _SETTING, setting2: Dict[str, Any]):
        for _, target_setting in setting1.items():
            update_keys = [key for key in target_setting.keys() if key in setting2]
            for key in update_keys:
                target_setting[key] = deepcopy(setting2[key])


class PruningSetting(ModuleSetting):
    default_setting = {
        'weight': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'granularity': 'default',
            'internal_metric_block': None,
            'apply_method': 'mul',
        },
        'bias': {
            'align': {
                'module_name': None,
                'target_name': 'weight',
                'dims': [0],
            },
            'apply_method': 'mul',
        }
    }

    embedding_setting = {
        'weight': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'granularity': 'default',
            'internal_metric_block': None,
            'apply_method': 'mul',
        }
    }

    # TODO: add more default module type
    registry = {
        'Linear': default_setting,
        'Embedding': embedding_setting,
        'Conv1d': default_setting,
        'Conv2d': default_setting,
        'Conv3d': default_setting,
        'ConvTranspose1d': default_setting,
        'ConvTranspose2d': default_setting,
        'ConvTranspose3d': default_setting,
        'BatchNorm1d': default_setting,
        'BatchNorm2d': default_setting,
        'BatchNorm3d': default_setting,
    }


class QuantizationSetting(ModuleSetting):
    default_setting = {
        '_input_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        'weight': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        # TODO: in which case should bias be quantized? need to check
        'bias': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        '_output_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        }
    }

    activation_setting = {
        '_input_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        '_output_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        }
    }

    # TODO: add more default module type
    registry = {
        'Linear': default_setting,
        'Conv1d': default_setting,
        'Conv2d': default_setting,
        'Conv3d': default_setting,
        'ConvTranspose1d': default_setting,
        'ConvTranspose2d': default_setting,
        'ConvTranspose3d': default_setting,
        'ReLU': activation_setting,
        'ReLU6': activation_setting,
        'BatchNorm1d': default_setting,
        'Hardtanh': activation_setting,
        'MaxPool2d': activation_setting,
        'BatchNorm2d': default_setting,
    }


# NOTE: not stable, might be changed in following stages
class DistillatoinSetting(ModuleSetting):
    default_setting = {
        '_output_': {
            'lambda': None,
            'link': None,
            'apply_method': None,
        }
    }

    registry = {}

    @classmethod
    def get(cls, module_cls_or_name: str | torch.nn.Module | Type[torch.nn.Module], config: Dict[str, Any]) -> _SETTING:
        # auto register default setting for unknown module type
        module_cls_name = cls._get_module_cls_name(module_cls_or_name)
        if module_cls_name not in cls.registry:
            cls.register(module_cls_name, cls.default_setting)
        return super().get(module_cls_or_name, config)


def canonicalize_settings(module: torch.nn.Module | Type[torch.nn.Module] | str,
                          config: Dict[str, Any],
                          mode: Literal['pruning', 'quantization', 'distillation']) -> Dict[str, Dict[str, Any]]:
    if mode == 'pruning':
        return PruningSetting.get(module, config)
    if mode == 'quantization':
        return QuantizationSetting.get(module, config)
    if mode == 'distillation':
        return DistillatoinSetting.get(module, config)
