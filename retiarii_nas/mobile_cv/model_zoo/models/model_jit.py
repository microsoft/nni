#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet traced models

Example code to create the model:
    from mobile_cv.model_zoo.models.model_jit import model_jit
    model = model_jit("fbnet_c_i8f_int8_jit", pretrained=True)
    model.eval()

Full example code is available at `examples/run_fbnet_v2_jit_int8.py`.

Architectures with pretrained weights could be found in:
    mobile_cv/model_zoo/models/model_info/model_jit/*.json
"""

import torch

from mobile_cv.model_zoo.models import hub_utils, utils


def _load_pretrained_info():
    folder_name = utils.get_model_info_folder("model_jit")
    ret = utils.load_model_info_all(folder_name)
    return ret


PRETRAINED_MODELS = _load_pretrained_info()


def load_jit_model(arch_name, progress=True):
    assert (
        arch_name in PRETRAINED_MODELS
    ), f"Invalid arch {arch_name}, supported arch {PRETRAINED_MODELS.keys()}"
    model_info = PRETRAINED_MODELS[arch_name]
    model_path = model_info["model_path"]
    if model_path.startswith("https://"):
        model_path = hub_utils.download_file(model_path, progress=progress)
    model = torch.jit.load(model_path, map_location="cpu")
    model.model_info = model_info
    return model


def model_jit(arch_name, pretrained=False, progress=True, **kwargs):
    """
    Constructs a model from traced model file defined in `PRETRAINED_MODELS`

    Args:
        arch_name (str): Architecture name
        pretrained (bool): Not used
        progress (bool): Not used
    """
    model = load_jit_model(arch_name)
    return model
