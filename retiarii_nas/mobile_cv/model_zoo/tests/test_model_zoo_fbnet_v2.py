#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

import mobile_cv.lut.lib.pt.flops_utils as flops_utils
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet


class TestModelZooFBNetV2(unittest.TestCase):
    def test_fbnet_v2(self):
        load_pretrained = True
        for name in ["fbnet_cse", "dmasking_l2_hs"]:
            print(f"Testing {name}...")
            model = fbnet(name, pretrained=load_pretrained)
            res = model.arch_def.get("input_size", 224)
            print(f"Test res: {res}")
            data = torch.zeros([1, 3, res, res])
            out = model(data)
            self.assertEqual(out.size(), torch.Size([1, 1000]))

    def test_fbnet_arch_def(self):
        model_arch = {
            "blocks": [
                # [c, s, n]
                # stage 0
                [["conv_k3_hs", 16, 2, 1]],
                # stage 1
                [["ir_k3", 16, 2, 1]],
                # stage 2
                [["ir_k3", 24, 2, 1]],
                # stage 3
                [["ir_pool_hs", 24, 1, 1]],
            ]
        }

        model = fbnet(model_arch, pretrained=False, num_classes=8)
        data = torch.zeros([1, 3, 32, 32])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 8]))

    def test_fbnet_flops(self):
        for x in ["fbnet_a", "fbnet_cse", "dmasking_f1"]:
            print(f"model name: {x}")
            model = fbnet(x, pretrained=False)
            res = model.arch_def.get("input_size", 224)
            input = torch.zeros([1, 3, res, res])
            flops_utils.print_model_flops(model, input)


if __name__ == "__main__":
    unittest.main()
