# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

import torch

from nni.compression.distillation import (
    Adaptive1dLayerwiseDistiller,
    DynamicLayerwiseDistiller
)

from ..assets.device import device
from ..assets.simple_mnist import (
    create_lighting_evaluator,
    create_pytorch_evaluator,
)

from ..assets.common import create_model
from ..assets.simple_mnist import SimpleLightningModel, SimpleTorchModel, create_lighting_evaluator, create_pytorch_evaluator


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
@pytest.mark.parametrize('distil_type', ['adaptive1d', 'dynamic'])
def test_adaptive_distiller(model_type: str, distil_type: str):
    model, config_list_dict, dummy_input = create_model(model_type)
    config_list = config_list_dict['distillation']

    if model_type == 'lightning':
        teacher_model = SimpleLightningModel().to(device)
        evaluator = create_lighting_evaluator()
    elif model_type == 'pytorch':
        teacher_model = SimpleTorchModel().to(device)
        evaluator = create_pytorch_evaluator(model)

    def teacher_predict(batch, model):
        batch = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)
        return model(batch)

    if distil_type == 'adaptive1d':
        distiller = Adaptive1dLayerwiseDistiller(model, config_list, evaluator, teacher_model, teacher_predict)
        distiller.track_forward(dummy_input)
    elif distil_type == 'dynamic':
        distiller = DynamicLayerwiseDistiller(model, config_list, evaluator, teacher_model, teacher_predict)
    distiller.compress(100, None)
