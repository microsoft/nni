# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

import torch
import torchvision.models as models

from nni.common.concrete_trace_utils import concrete_trace

model_list = [
    models.alexnet,
    models.convnext_base,
    models.densenet121,
    models.efficientnet_b0,
    models.mobilenet_v2,
    models.resnet18,
    models.resnext50_32x4d,
    models.vit_b_16,
]


def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        # may not euqal on gpu
        return torch.std(a - b).item() < 1e-6
    else:
        return a == b

@pytest.mark.parametrize('model_fn', model_list)
def test_torchvision_models(model_fn):
    model = model_fn()
    model.eval()
    dummy_inputs = (torch.rand(2, 3, 224, 224), )
    traced = concrete_trace(model, dummy_inputs, use_operator_patch=True)
    out_orig = model.forward(*dummy_inputs)
    out_traced = traced.forward(*dummy_inputs)
    assert check_equal(out_orig, out_traced), f'{traced.code}'
    del out_orig, out_traced