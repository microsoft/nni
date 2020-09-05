#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from mobile_cv.common.misc.registry import Registry


def test_func():
    return "test_func"


class TestRegistry(unittest.TestCase):
    def test_registry(self):
        REG_LIST = Registry("test")
        REG_LIST.register("test_func", test_func)

        @REG_LIST.register()
        def test_func1():
            return "test_func_1"

        self.assertEqual(len(REG_LIST.get_names()), 2)

        out = REG_LIST.get("test_func")()
        self.assertEqual(out, "test_func")
        out1 = REG_LIST.get("test_func1")()
        self.assertEqual(out1, "test_func_1")
