#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy

import torch
import torch.nn as nn

import mobile_cv.arch.fbnet_v2.basic_blocks as bb


def fuse_convbnrelu(module, inplace=False):
    ret = module if inplace else copy.deepcopy(module)
    if isinstance(module, bb.ConvBNRelu):
        SUPPORTED_TYPES = [
            ("conv", [nn.Conv2d]),
            ("bn", [nn.BatchNorm2d]),
            ("relu", [nn.ReLU]),
        ]

        def _is_supported(module, name, supported_types):
            op = getattr(module, name, None)
            if op is None:
                return False
            if type(op) not in supported_types:
                return False
            return True

        names = [
            name
            for name, sp_types in SUPPORTED_TYPES
            if _is_supported(module, name, sp_types)
        ]
        assert len(names) > 0
        if len(names) > 1:
            ret = torch.quantization.fuse_modules(
                module, names, inplace=inplace
            )
    return ret


def fuse_model(model: nn.Module, inplace=False):
    model = fuse_convbnrelu(model, inplace=inplace)
    children = {}
    for name, child in model.named_children():
        children[name] = fuse_model(child, inplace=inplace)
    if not inplace:
        for name, child in children.items():
            setattr(model, name, child)
    return model


def check_bn_exist(model):
    for x in model.modules():
        if isinstance(x, nn.BatchNorm2d):
            return True
    return False
