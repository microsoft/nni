# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import torch

from .device import device
from .simple_mnist import SimpleLightningModel, SimpleTorchModel


log_dir = Path(__file__).parent.parent / 'logs'


def create_model(model_type: str):
    torch_config_list = [{'op_types': ['Linear'], 'sparsity': 0.75},
                         {'op_names': ['conv1', 'conv2', 'conv3'], 'sparsity': 0.75},
                         {'op_names': ['fc2'], 'exclude': True}]

    lightning_config_list = [{'op_types': ['Linear'], 'sparsity': 0.75},
                             {'op_names': ['model.conv1', 'model.conv2', 'model.conv3'], 'sparsity': 0.75},
                             {'op_names': ['model.fc2'], 'exclude': True}]

    if model_type == 'lightning':
        model = SimpleLightningModel()
        config_list = lightning_config_list
        dummy_input = torch.rand(8, 1, 28, 28)
    elif model_type == 'pytorch':
        model = SimpleTorchModel().to(device)
        config_list = torch_config_list
        dummy_input = torch.rand(8, 1, 28, 28, device=device)
    else:
        raise ValueError(f'wrong model_type: {model_type}')
    return model, config_list, dummy_input
