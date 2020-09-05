#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import unittest

from mobile_cv.common.misc.file_utils import make_temp_directory
from mobile_cv.common.misc.imports import import_file


class TestMisc(unittest.TestCase):
    def test_import_from_file(self):
        with make_temp_directory("test_import_from_file") as dir:
            filename = os.path.join(dir, "my_module.py")
            with open(filename, "w") as f:
                f.write("ANSWER = 42\n")

            my_module = import_file("my_module", filename)
            self.assertEqual(my_module.ANSWER, 42)
