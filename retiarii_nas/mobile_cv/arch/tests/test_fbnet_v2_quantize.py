#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder
import mobile_cv.arch.utils.fuse_utils as fuse_utils


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


class TestFBNetV2Quantize(unittest.TestCase):
    def test_post_quant(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 4, 2, 1, bn_args)],
                # stage 1
                [
                    ("ir_k3", 8, 2, 2, e6, dw_skip_bnrelu, bn_args),
                    ("ir_k5", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)
        model = torch.quantization.QuantWrapper(model)
        model = fuse_utils.fuse_model(model, inplace=False)

        print(f"Fused model {model}")

        model.qconfig = torch.quantization.default_qconfig
        print(model.qconfig)
        torch.quantization.prepare(model, inplace=True)

        # calibration
        for _ in range(5):
            data = torch.rand([2, 3, 8, 8])
            model(data)

        # Convert to quantized model
        quant_model = torch.quantization.convert(model, inplace=False)
        print(f"Quant model {quant_model}")

        quant_output = quant_model(torch.rand([2, 3, 8, 8]))

        # Make sure model can be traced
        torch.jit.trace(quant_model, torch.randn([2, 3, 8, 8]))

        self.assertEqual(quant_output.shape, torch.Size([2, 8, 2, 2]))
