#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import numpy as np
import torch

import mobile_cv.arch.fbnet_v2.basic_blocks as bb
import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils


def run_and_compare(model_before, model_after, input_size):
    inputs = torch.zeros(input_size, requires_grad=False)
    output_before = model_before(inputs)
    output_after = model_after(inputs)

    np.testing.assert_allclose(
        output_before.detach(), output_after.detach(), rtol=0, atol=1e-4
    )


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


def _find_modules(model, module_to_check):
    for x in model.modules():
        if isinstance(x, module_to_check):
            return True
    return False


class TestUtilsFuseUtils(unittest.TestCase):
    def test_fuse_convbnrelu(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=False)

        self.assertTrue(_find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_convbnrelu_inplace(self):
        cbr = bb.ConvBNRelu(
            3, 6, kernel_size=3, padding=1, bn_args="bn", relu_args="relu"
        ).eval()
        fused = fuse_utils.fuse_convbnrelu(cbr, inplace=True)

        self.assertFalse(_find_modules(cbr, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused, torch.nn.BatchNorm2d))

        input_size = [2, 3, 7, 7]
        run_and_compare(cbr, fused, input_size)

    def test_fuse_model(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=False)
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))

        input_size = [2, 3, 8, 8]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_inplace(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=True)
        print(model)
        print(fused_model)

        self.assertFalse(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))

        input_size = [2, 3, 8, 8]
        run_and_compare(model, fused_model, input_size)

    def test_fuse_model_swish(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args, {"relu_args": "swish"})],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        fused_model = fuse_utils.fuse_model(model, inplace=False)
        print(model)
        print(fused_model)

        self.assertTrue(_find_modules(model, torch.nn.BatchNorm2d))
        self.assertFalse(_find_modules(fused_model, torch.nn.BatchNorm2d))
        self.assertTrue(_find_modules(fused_model, bb.Swish))

        input_size = [2, 3, 8, 8]
        run_and_compare(model, fused_model, input_size)
