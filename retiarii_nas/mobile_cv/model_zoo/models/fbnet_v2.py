#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet classification models

Example code to create the model:
    from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
    model = fbnet("fbnet_cse", pretrained=True)
    model.eval()

Full example code is available at `examples/run_fbnet_v2.py`.

All suported architectures could be found in:
    mobile_cv/arch/fbnet_v2/fbnet_modeldef_cls*.py

Architectures with pretrained weights could be found in:
    mobile_cv/model_zoo/models/model_info/fbnet_v2/*.json
"""

import typing

import torch
import torch.nn as nn

from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder
from mobile_cv.arch.fbnet_v2 import fbnet_modeldef_cls as modeldef
from mobile_cv.arch.utils import misc
from mobile_cv.model_zoo.models import hub_utils, utils


def _load_pretrained_info():
    folder_name = utils.get_model_info_folder("fbnet_v2")
    ret = utils.load_model_info_all(folder_name)
    return ret


PRETRAINED_MODELS = _load_pretrained_info()

NAME_MAPPING = {
    # external name : internal name
    "FBNet_a": "fbnet_a",
    "FBNet_b": "fbnet_b",
    "FBNet_c": "fbnet_c",
    "FBNet_ase": "fbnet_ase",
    "FBNet_bse": "fbnet_ase",
    "FBNet_cse": "fbnet_ase",
    "MobileNetV3": "mnv3",
    "FBNetV2_F1": "dmasking_f1",
    "FBNetV2_F5": "dmasking_l2",
}


def _load_fbnet_state_dict(file_name, progress=True):
    if file_name.startswith("https://"):
        file_name = hub_utils.download_file(file_name, progress=progress)

    state_dict = torch.load(file_name, map_location="cpu")
    if "model_ema" in state_dict and state_dict["model_ema"] is not None:
        state_dict = state_dict["model_ema"]
    else:
        state_dict = state_dict["state_dict"]
    ret = {}
    for name, val in state_dict.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        ret[name] = val
    return ret


def _create_builder(arch_name_or_def: typing.Union[str, dict]):
    if isinstance(arch_name_or_def, str):
        assert arch_name_or_def in modeldef.MODEL_ARCH, (
            f"Invalid arch name {arch_name_or_def}, "
            f"available names: {modeldef.MODEL_ARCH.keys()}"
        )
        arch_def = modeldef.MODEL_ARCH[arch_name_or_def]
    else:
        assert isinstance(arch_name_or_def, dict)
        arch_def = arch_name_or_def

    arch_def = mbuilder.unify_arch_def(arch_def, ["blocks"])

    scale_factor = 1.0
    width_divisor = 1
    bn_info = {"name": "bn", "momentum": 0.003}
    drop_out = 0.2

    arch_def["dropout_ratio"] = drop_out

    builder = mbuilder.FBNetBuilder(
        width_ratio=scale_factor, bn_args=bn_info, width_divisor=width_divisor
    )
    builder.add_basic_args(**arch_def.get("basic_args", {}))

    return builder, arch_def


class ClsConvHead(nn.Module):
    """Global average pooling + conv head for classification
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # global avg pool of arbitrary feature map size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


class FBNetBackbone(nn.Module):
    def __init__(self, arch_name, dim_in=3):
        super().__init__()

        builder, arch_def = _create_builder(arch_name)

        self.stages = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
        self.dropout = misc.add_dropout(arch_def["dropout_ratio"])
        self.out_channels = builder.last_depth
        self.arch_def = arch_def

    def forward(self, x):
        y = self.stages(x)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class FBNet(nn.Module):
    def __init__(self, arch_name, dim_in=3, num_classes=1000):
        super().__init__()
        self.backbone = FBNetBackbone(arch_name, dim_in)
        self.head = ClsConvHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.head(y)
        return y

    @property
    def arch_def(self):
        return self.backbone.arch_def


def fbnet(arch_name, pretrained=False, progress=True, **kwargs):
    """
    Constructs a FBNet architecture named `arch_name`

    Args:
        arch_name (str): Architecture name
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if isinstance(arch_name, str) and arch_name in NAME_MAPPING:
        arch_name = NAME_MAPPING[arch_name]

    model = FBNet(arch_name, **kwargs)
    if pretrained:
        assert (
            arch_name in PRETRAINED_MODELS
        ), f"Invalid arch {arch_name}, supported arch {PRETRAINED_MODELS.keys()}"
        model_info = PRETRAINED_MODELS[arch_name]
        model_path = model_info["model_path"]
        state_dict = _load_fbnet_state_dict(model_path, progress=progress)
        model.load_state_dict(state_dict)
        model.model_info = model_info
    return model
