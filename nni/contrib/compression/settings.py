from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import torch

INPUT_PREFIX = '_input_'
OUTPUT_PREFIX = '_output_'


class ModuleSetting:
    registry: Dict[str, Dict[str, Any]]

    @classmethod
    def get(cls, module: torch.nn.Module, config: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
        module_type = module.__class__.__name__
        module_setting = deepcopy(cls.registry.get(module_type, None))
        customized_setting = deepcopy({} if config is None else config.get('advance', {}))
        assert module_setting is not None or ('target_names' in customized_setting and 'target_settings' in customized_setting), \
            f'{module_type} is not registered, please set `target_names` and `target_settings` in config.'

        target_names: List[str] = customized_setting.get('target_names', module_setting.get('target_names'))
        target_settings = customized_setting.get('target_settings', module_setting.get('target_settings'))

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

        if config is not None:
            for _, target_setting in selected_settings.items():
                for key, val in config.items():
                    if key == 'advance':
                        continue
                    if key in target_setting:
                        target_setting[key] = val
        return selected_settings


class PruningSetting(ModuleSetting):
    default_setting = {
        'target_names': ['weight', 'bias'],
        'target_settings': {
            'weight': {
                'sparsity_ratio': None,
                'sparsity_threshold': None,
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
    }

    registry = {
        'Conv2d': default_setting,
        'Linear': default_setting
    }


class QuantizationSetting(ModuleSetting):
    default_setting = {
        'target_names': ['_input_', 'weight', 'bias', '_output_'],
        'target_settings': {
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
    }

    registry = {
        'Conv2d': default_setting,
        'Linear': default_setting
    }


class DistillatoinSetting(ModuleSetting):
    pass
