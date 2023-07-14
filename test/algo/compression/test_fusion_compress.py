# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from nni.compression.distillation import (
    DynamicLayerwiseDistiller,
    Adaptive1dLayerwiseDistiller
)
from nni.compression.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    TaylorPruner,
    MovementPruner,
    LinearPruner,
    AGPPruner,
)
from nni.compression.quantization import (
    BNNQuantizer,
    DoReFaQuantizer,
    LsqQuantizer,
    PtqQuantizer,
    QATQuantizer,
)

from nni.compression.speedup import ModelSpeedup

from .assets.common import create_model
from .assets.device import device
from .assets.simple_mnist import SimpleLightningModel, SimpleTorchModel, create_lighting_evaluator, create_pytorch_evaluator


@pytest.mark.parametrize('model_type', ['lightning', 'pytorch'])
def test_fusion_compress(model_type: str):
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

    config_list = config_list_dict['distillation']
    distiller = DynamicLayerwiseDistiller(model, config_list, evaluator, teacher_model, teacher_predict)

    config_list = config_list_dict['pruning']
    pruner = TaylorPruner.from_compressor(distiller, config_list, 10)

    config_list = config_list_dict['quantization']
    quantizer = LsqQuantizer.from_compressor(pruner, config_list)

    quantizer.compress(100, None)
