# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pytest
import torch
import torchvision.models as tm
from nni.common.concrete_trace_utils import concrete_trace
from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup, auto_set_denpendency_group_ids

models = [
    tm.alexnet,
    # tm.convnext_tiny,
    tm.densenet121,
    tm.efficientnet_b0,
    tm.inception_v3,
    tm.mnasnet0_5,
    tm.mobilenet_v2,
    tm.resnet18,
    tm.resnext50_32x4d,
    # tm.shufflenet_v2_x0_5,
    tm.squeezenet1_0,
    tm.vgg11,
    tm.wide_resnet50_2,
]


@pytest.mark.parametrize('model_fn', models)
def test_pruner_speedup(model_fn):
    model = model_fn()
    dummy_inputs = (torch.rand(2, 3, 224, 224), )
    
    config_list = [{
        'op_types': ['Conv2d'],
        'sparsity': 0.5
    }]
    traced = concrete_trace(model, dummy_inputs, use_operator_patch=True)
    config_list = auto_set_denpendency_group_ids(traced, config_list)
    
    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()
    pruner.unwrap_model()

    ModelSpeedup(model, dummy_inputs, masks, graph_module=traced).speedup_model()
    traced.forward(*dummy_inputs)
    

if __name__ == '__main__':
    test_pruner_speedup(tm.shufflenet_v2_x0_5)