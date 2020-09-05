#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.arch.fbnet_v2.fbnet_builder as mbuilder


class TestFBNetV2BuilderUtils(unittest.TestCase):
    def test_unify_arch(self):
        e6 = {"expansion": 6}
        dw_skip_bnrelu = {"dw_skip_bnrelu": True}
        arch_def = {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [
                    ("ir_k3", 64, 2, 2, e6, dw_skip_bnrelu),
                    ("ir_k5", 96, 1, 1, e6),
                ],
            ],
            "backbone": [0],
            "heads": [1],
        }

        gt_unified_arch = {
            "blocks": [
                {
                    "stage_idx": 0,
                    "block_idx": 0,
                    "block_op": "conv_k3",
                    "block_cfg": {"out_channels": 32, "stride": 2},
                },
                {
                    "stage_idx": 1,
                    "block_idx": 0,
                    "block_op": "ir_k3",
                    "block_cfg": {
                        "out_channels": 64,
                        "stride": 2,
                        "expansion": 6,
                        "dw_skip_bnrelu": True,
                    },
                },
                {
                    "stage_idx": 1,
                    "block_idx": 1,
                    "block_op": "ir_k3",
                    "block_cfg": {
                        "out_channels": 64,
                        "stride": 1,
                        "expansion": 6,
                        "dw_skip_bnrelu": True,
                    },
                },
                {
                    "stage_idx": 1,
                    "block_idx": 2,
                    "block_op": "ir_k5",
                    "block_cfg": {
                        "out_channels": 96,
                        "stride": 1,
                        "expansion": 6,
                    },
                },
            ],
            "backbone": [0],
            "heads": [1],
        }

        unified_arch = mbuilder.unify_arch_def(arch_def, ["blocks"])
        self.assertEqual(unified_arch, gt_unified_arch)

    def test_count_strides(self):
        e6 = {"expansion": 6}
        arch_def = {
            "blocks1": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 64, 2, 2, e6), ("ir_k5", 96, 1, 1, e6)],
            ],
            "blocks2": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 64, -2, 2, e6), ("ir_k5", 96, -2, 1, e6)],
            ],
        }

        unified_arch = mbuilder.unify_arch_def(arch_def, ["blocks1", "blocks2"])

        gt_strides_blocks1 = [2, 2, 1, 1]
        gt_strides_blocks2 = [2, 0.5, 1, 0.5]
        count_strides1 = mbuilder.count_stride_each_block(
            unified_arch["blocks1"]
        )
        count_strides2 = mbuilder.count_stride_each_block(
            unified_arch["blocks2"]
        )
        self.assertEqual(gt_strides_blocks1, count_strides1)
        self.assertEqual(gt_strides_blocks2, count_strides2)

        all_strides1 = mbuilder.count_strides(unified_arch["blocks1"])
        all_strides2 = mbuilder.count_strides(unified_arch["blocks2"])
        self.assertEqual(all_strides1, 4)
        self.assertEqual(all_strides2, 0.5)

    def test_count_stages(self):
        e6 = {"expansion": 6}
        arch_def = {
            "blocks1": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 64, 2, 2, e6), ("ir_k5", 96, 1, 1, e6)],
                # stage 2
                [("ir_k3", 64, -2, 2, e6), ("ir_k5", 96, -2, 1, e6)],
            ]
        }
        arch_def = mbuilder.unify_arch_def(arch_def, ["blocks1"])

        num_stages = mbuilder.get_num_stages(arch_def["blocks1"])
        self.assertEqual(num_stages, 3)

        gt_stage_out_channels = [32, 96, 96]
        stage_out_channels = mbuilder.get_stages_dim_out(arch_def["blocks1"])
        self.assertEqual(stage_out_channels, gt_stage_out_channels)

        gt_blocks_in_stage = [1, 3, 3]
        blocks_in_stage = mbuilder.get_num_blocks_in_stage(arch_def["blocks1"])
        self.assertEqual(blocks_in_stage, gt_blocks_in_stage)


if __name__ == "__main__":
    unittest.main()
