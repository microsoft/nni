#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

import mobile_cv.arch.utils.misc as misc


class TestUtilsMisc(unittest.TestCase):
    def test_misc_drop_connect_batch(self):
        N, C, H, W = 500, 2, 3, 3
        drop_rate = 0.2
        inputs = torch.ones((N, C, H, W), dtype=torch.float32)
        output = misc.drop_connect_batch(inputs, drop_rate, training=True)
        total_count = torch.sum(inputs).item()
        zeroed_count = torch.sum(output == 0.0).item()
        actual_drop_rate = zeroed_count / total_count

        self.assertEqual(inputs.shape, output.shape)
        self.assertAlmostEqual(actual_drop_rate, drop_rate, 1)
