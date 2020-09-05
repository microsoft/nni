#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder


def _build_model(arch_def, dim_in):
    arch_def = fbnet_builder.unify_arch_def(arch_def, ["blocks"])
    torch.manual_seed(0)
    builder = fbnet_builder.FBNetBuilder(1.0)
    model = builder.build_blocks(arch_def["blocks"], dim_in=dim_in)
    model.eval()
    return model


def _get_input(n, c, h, w):
    nchw = n * c * h * w
    input = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
    input = input.reshape(n, c, h, w)
    return input


class TestFBNetBuilder(unittest.TestCase):
    def test_fbnet_builder_check_output(self):
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
                    ("ir_k5_sehsig", 8, 1, 1, e6, bn_args),
                ],
            ]
        }

        model = _build_model(arch_def, dim_in=3)

        print(model)

        input = _get_input(2, 3, 8, 8)
        output = model(input)
        self.assertEqual(output.shape, torch.Size([2, 8, 2, 2]))


if __name__ == "__main__":
    unittest.main()
