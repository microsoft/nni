#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

import torch

from mobile_cv.model_zoo.models.model_jit import model_jit


class TestModelZooModelJit(unittest.TestCase):
    def test_fbnet_v2_jit_int8(self):
        model = model_jit("fbnet_c_i8f_int8_jit")
        data = torch.zeros([1, 3, 224, 224])
        out = model(data)
        self.assertEqual(out.size(), torch.Size([1, 1000]))


if __name__ == "__main__":
    unittest.main()
