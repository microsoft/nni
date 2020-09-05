#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest

import torch
import torch.nn as nn

import mobile_cv.lut.lib.lut_ops as lut_ops
import mobile_cv.lut.lib.pt.utils as pt_utils
from mobile_cv.lut.lib.lut_schema import OpInfo


class EmptyJitScriptModule(torch.jit.ScriptModule):
    def forward(self, x):
        return x


def create_model():
    ret = nn.Sequential(
        nn.Conv2d(
            3,
            8,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        nn.ReLU(),
        nn.Conv2d(
            8,
            8,
            3,
            stride=(2, 2),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        nn.ConvTranspose2d(
            8,
            4,
            3,
            stride=(1, 1),
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=False,
        ),
        EmptyJitScriptModule(),
        nn.ReLU(),
    )
    ret.OP_INPUT = [2, 3, 16, 16]
    return ret


class TestPTUtils(unittest.TestCase):
    def test_module_hook(self):
        model = create_model()
        input = torch.zeros(model.OP_INPUT)

        def callback(m, input, output):
            nonlocal count
            count += 1

        mc = pt_utils.ModuleHook(callback).register_forward_hook(model)
        count = 0
        for _ in range(5):
            model(input)
        self.assertEqual(count, 5)
        mc.remove_hook()
        for _ in range(2):
            model(input)
        self.assertEqual(count, 5)

        with pt_utils.ModuleHook(callback).register_forward_hook(model):
            count = 0
            for _ in range(4):
                model(input)
            self.assertEqual(count, 4)
        for _ in range(2):
            model(input)
        self.assertEqual(count, 4)

        with pt_utils.ModuleHook(callback, 3).register_forward_hook(model):
            count = 0
            for _ in range(4):
                model(input)
            self.assertEqual(count, 3)

        pt_utils.ModuleHook(callback, 3).register_forward_hook(model)
        count = 0
        for _ in range(4):
            model(input)
        self.assertEqual(count, 3)

    def test_nested_hook(self):
        model = create_model()
        input = torch.zeros(model.OP_INPUT)

        counts = []

        def _hook(m, input, output):
            counts.append(1)
            return {"name": m.__class__.__name__}

        def _get_check_result(data):
            def _check_hook_output(m):
                if len(list(m.children())) > 0:
                    return
                self.assertEqual(data(m, "name"), m.__class__.__name__)

            return _check_hook_output

        with pt_utils.NestedModuleHook(_hook).register_forward_hook(
            model
        ) as data:
            model(input)
            model.apply(_get_check_result(data))
        self.assertEqual(sum(counts), 5)

        # make sure the hooks are removed
        counts = []
        model(input)
        self.assertEqual(counts, [])

    def test_convert_lut_ops(self):
        model = create_model()
        op_input = copy.deepcopy(model.OP_INPUT)

        ops = pt_utils.convert_to_lut_ops(model, op_input)
        self.assertEqual(len(ops), 3)

        lut_ops_gt = [
            OpInfo(
                lut_ops.Conv2d(
                    3,
                    8,
                    3,
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=1,
                    groups=1,
                    bias=False,
                ),
                input_shapes=[[2, 3, 16, 16]],
            ),
            OpInfo(
                lut_ops.Conv2d(
                    8,
                    8,
                    3,
                    stride=(2, 2),
                    padding=(1, 1),
                    dilation=1,
                    groups=1,
                    bias=False,
                ),
                input_shapes=[[2, 8, 16, 16]],
            ),
            OpInfo(
                lut_ops.ConvTranspose2d(
                    8,
                    4,
                    3,
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=1,
                    groups=1,
                    bias=False,
                ),
                input_shapes=[[2, 8, 8, 8]],
            ),
        ]
        self.assertEqual(lut_ops_gt, ops)


if __name__ == "__main__":
    unittest.main()
