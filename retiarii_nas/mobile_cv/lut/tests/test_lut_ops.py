#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import mobile_cv.lut.lib.lut_ops as lut_ops


class TestLutOps(unittest.TestCase):
    def test_op_conv(self):
        op1 = lut_ops.Conv2d(
            3,
            8,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        )
        op1_input = [[2, 3, 224, 224]]

        op2 = lut_ops.Conv2d(8, 4, 3, stride=1, padding=1, groups=4)
        op2_input = op1.get_output_shape(op1_input)

        nparams = op1.get_nparams() + op2.get_nparams()
        self.assertAlmostEqual(nparams, 288)

        nflops = op1.get_flops(op1_input) + op2.get_flops(op2_input)
        self.assertAlmostEqual(nflops, 14450688 * 2)

    def test_op_conv_shape(self):
        op1 = lut_ops.Conv2d(
            4,
            8,
            (5, 3),
            stride=(1, 2),
            padding=(2, 4),
            dilation=3,
            groups=4,
            bias=False,
        )
        op1_input = [[2, 4, 224, 224]]

        op1_output = op1.get_output_shape(op1_input)
        self.assertEqual(op1_output, [[2, 8, 216, 113]])

        op1_params_shape = op1.get_params_shape()
        self.assertEqual(op1_params_shape, [[8, 1, 5, 3]])

    def test_op_conv_transpose(self):
        op1 = lut_ops.ConvTranspose2d(
            3,
            8,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        )
        op1_input = [[2, 3, 224, 224]]

        nparams = op1.get_nparams()
        self.assertAlmostEqual(nparams, 216)

        nflops = op1.get_flops(op1_input)
        self.assertAlmostEqual(nflops, 10838016 * 2)

    def test_op_conv_transpose_shape(self):
        op1 = lut_ops.ConvTranspose2d(
            4,
            8,
            (5, 3),
            stride=(2, 2),
            padding=(2, 4),
            output_padding=(2, 1),
            dilation=3,
            groups=4,
            bias=False,
        )
        op1_input = [[2, 4, 224, 224]]

        op1_output = op1.get_output_shape(op1_input)
        self.assertEqual(op1_output, [[2, 8, 457, 446]])

        op1_params_shape = op1.get_params_shape()
        self.assertEqual(op1_params_shape, [[4, 2, 5, 3]])

    def test_op_linear(self):
        op1 = lut_ops.Linear(3, 8)
        op1_input = [[2, 3]]

        op2 = lut_ops.Linear(8, 12)
        op2_input = op1.get_output_shape(op1_input)

        nparams = op1.get_nparams() + op2.get_nparams()
        self.assertAlmostEqual(nparams, 120)

        nflops = op1.get_flops(op1_input) + op2.get_flops(op2_input)
        self.assertAlmostEqual(nflops, 120 * 2)

    def test_adaptive_avg_pool(self):
        shape = (1, 3, 500, 500)
        op = lut_ops.AdaptiveAvgPool2d()
        self.assertEqual(op.get_flops(shape), 750000)
        self.assertEqual(op.get_nparams(), 0)


if __name__ == "__main__":
    unittest.main()
