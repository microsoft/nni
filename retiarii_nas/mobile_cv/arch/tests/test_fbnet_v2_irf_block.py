#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

import mobile_cv.arch.fbnet_v2.irf_block as irf_block

TEST_CUDA = torch.cuda.is_available()


def create_test_irf(self, out_channels, op_args, input_shape, gt_output_dim):
    N, C_in, H, W = input_shape
    op = irf_block.IRFBlock(
        in_channels=C_in, out_channels=out_channels, **op_args
    )
    print(op)

    input = torch.rand(input_shape, dtype=torch.float32)
    output = op(input)

    self.assertEqual(
        output.shape,
        torch.Size([N, out_channels, gt_output_dim, gt_output_dim]),
    )


class TestIRFBlocks(unittest.TestCase):
    def test_irf_block(self):
        N, C_in, C_out = 2, 16, 16
        input_dim = 7

        for bn_args in [
            "bn",
            {"name": "bn", "momentum": 0.003},
            {"name": "sync_bn", "momentum": 0.003},
        ]:
            with self.subTest(f"bn={bn_args}"):
                create_test_irf(
                    self,
                    C_out,
                    op_args={
                        "expansion": 6,
                        "kernel_size": 3,
                        "stride": 1,
                        "bn_args": bn_args,
                    },
                    input_shape=[N, C_in, input_dim, input_dim],
                    gt_output_dim=input_dim,
                )

        with self.subTest(f"skip_bnrelu=True"):
            create_test_irf(
                self,
                C_out,
                op_args={
                    "expansion": 6,
                    "kernel_size": 3,
                    "stride": 1,
                    "bn_args": "bn",
                    "dw_skip_bnrelu": True,
                },
                input_shape=[N, C_in, input_dim, input_dim],
                gt_output_dim=input_dim,
            )

    def test_irf_block_res_conn(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 1},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_irf(
            self,
            C_out,
            op_args={"expansion": 6, "kernel_size": 3, "stride": 2},
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_irf_block_se(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 1,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim,
        )

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": 2,
                "se_args": {"name": "se_hsig", "relu_args": "hswish"},
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim // 2,
        )

    def test_irf_block_upsample(self):
        N, C_in, C_out = 2, 16, 32
        input_dim = 8

        create_test_irf(
            self,
            C_out,
            op_args={
                "expansion": 6,
                "kernel_size": 3,
                "stride": -2,
                "se_args": "se_fc",
            },
            input_shape=[N, C_in, input_dim, input_dim],
            gt_output_dim=input_dim * 2,
        )
