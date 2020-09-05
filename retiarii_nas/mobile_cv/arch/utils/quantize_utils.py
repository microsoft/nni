#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing

import torch

from . import fuse_utils


def quantize_model(
    model_builder: typing.Callable,
    inputs,
    add_quant_stub=True,
    quant_config=None,
):
    print("Building quantization compatiable model...")

    model = model_builder()
    if add_quant_stub:
        model = torch.quantization.QuantWrapper(model)

    print("Fusing bn...")
    model = fuse_utils.fuse_model(model)
    assert not fuse_utils.check_bn_exist(model), model

    model.qconfig = quant_config or torch.quantization.default_qconfig
    print(f"Quant config: {model.qconfig}")

    torch.quantization.prepare(model, inplace=True)
    print("Collecting stats...")
    model(inputs)
    quant_model = torch.quantization.convert(model, inplace=False)

    return quant_model
