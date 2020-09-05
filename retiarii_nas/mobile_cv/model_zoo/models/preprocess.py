#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Preprocess function for pretrained models
"""

import math

from torchvision import transforms


def get_preprocess(crop_res, resize_res=0):
    if resize_res <= 0:
        ratio = 256.0 / 224.0
        resize_res = int(math.ceil(ratio * crop_res / 8) * 8.0)
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_res),
            transforms.CenterCrop(crop_res),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return preprocess


def get_preprocess_from_model_info(model_info):
    assert "resolution" in model_info, model_info
    pargs = {
        "crop_res": model_info["resolution"],
        "resize_res": model_info.get("resize", 0),
    }
    return get_preprocess(**pargs)
