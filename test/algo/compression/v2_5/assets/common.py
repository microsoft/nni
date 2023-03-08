# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import torch

from .device import device
from .simple_mnist import SimpleLightningModel, SimpleTorchModel


log_dir = Path(__file__).parent.parent / 'logs'


def create_model(model_type: str):
    pruning_torch_config_list = [{'op_types': ['Linear'], 'exclude_op_names': ['fc2'], 'sparse_ratio': 0.75},
                                 {'op_names': ['conv1', 'conv2', 'conv3'], 'sparse_ratio': 0.75}]

    distil_torch_config_list = [{'op_types': ['Linear'], 'lambda': 0.75},
                                {'op_names': ['conv1', 'conv2', 'conv3'], 'lambda': 0.75}]

    pruning_lightning_config_list = [{'op_types': ['Linear'], 'exclude_op_names': ['model.fc2'], 'sparse_ratio': 0.75},
                                     {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'sparse_ratio': 0.75}]

    distil_lightning_config_list = [{'op_types': ['Linear'], 'lambda': 0.75},
                                    {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'lambda': 0.75}]


    if model_type == 'lightning':
        model = SimpleLightningModel().to(device)
        config_list_dict = {
            'pruning': pruning_lightning_config_list,
            'distillation': distil_lightning_config_list
        }
        dummy_input = torch.rand(8, 1, 28, 28, device=device)
    elif model_type == 'pytorch':
        model = SimpleTorchModel().to(device)
        config_list_dict = {
            'pruning': pruning_torch_config_list,
            'distillation': distil_torch_config_list
        }
        dummy_input = torch.rand(8, 1, 28, 28, device=device)
    else:
        raise ValueError(f'wrong model_type: {model_type}')
    return model, config_list_dict, dummy_input
