#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import mobile_cv.lut.lib.lut_ops as lut_ops
import mobile_cv.lut.lib.lut_schema as lut


class TestLutSchema(unittest.TestCase):
    def test_lut_table(self):
        item1 = lut.LutItem(
            op=lut_ops.Conv2d(
                3,
                20,
                3,
                stride=(2, 2),
                padding=(1, 1),
                dilation=1,
                groups=1,
                bias=False,
            ),
            input_shapes=[[1, 3, 224, 224]],
        )
        item2 = lut.LutItem(
            op=lut_ops.Conv2d(
                20,
                20,
                5,
                stride=1,
                padding=0,
                dilation=(1, 1),
                groups=2,
                bias=False,
            ),
            input_shapes=[[1, 20, 224, 224]],
        )
        item3 = lut.LutItem(
            op=lut_ops.ConvTranspose2d(
                20,
                20,
                5,
                stride=2,
                padding=1,
                output_padding=1,
                dilation=(1, 1),
                groups=2,
                bias=False,
            ),
            input_shapes=[[1, 20, 224, 224]],
        )

        table = lut.LutTable()
        table.extend([item1, item2, item3])

        fd, path = tempfile.mkstemp()
        try:
            table.save(path)
            table_loaded = lut.LutTable.Load(path)
            assert table == table_loaded

        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
