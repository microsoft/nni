# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

import torch

from nni.compression.base.setting import canonicalize_settings


@pytest.mark.parametrize("module_cls_or_name", ['instance', 'cls', 'name'])
def test_setting(module_cls_or_name: str):
    if module_cls_or_name == 'instance':
        module = torch.nn.Linear(1, 1)
    elif module_cls_or_name == 'cls':
        module = torch.nn.Linear
    elif module_cls_or_name == 'name':
        module = 'Linear'

    config = {
        'target_names': ['weight', 'bias'],
        # shortcut
        'sparse_ratio': 0.5,
        'target_settings': {
            'bias': {
                # customized key
                'sparse_ratio': 0.6
            }
        }
    }

    setting = canonicalize_settings(module, config, mode='pruning')
    assert setting['weight']['sparse_ratio'] == 0.5 and setting['bias']['sparse_ratio'] == 0.6

    config = {
        'target_names': ['_input_', 'weight', '_output_'],
        # shortcut
        'quant_dtype': 'uint6',
        # not exist shortcut will not appear in setting
        'sparse_ratio': 0.4,
        'target_settings': {
            'weight': {
                # customized key
                'sparse_ratio': 0.6
            }
        }
    }

    setting = canonicalize_settings(module, config, mode='quantization')
    assert 'sparse_ratio' not in setting['_input_0']
    assert setting['weight']['quant_dtype'] == 'uint6' and setting['weight']['sparse_ratio'] == 0.6

    config = {
        'target_names': ['_output_'],
        'lambda': 0.5
    }
    # Linear not in distillation registry, distillation setting should auto-register Linear with default setting
    setting = canonicalize_settings(module, config, mode='distillation')
    assert setting['_output_0']['lambda'] == 0.5
