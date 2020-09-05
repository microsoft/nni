#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import unittest

import numpy as np
import torch

import mobile_cv.arch.fbnet_v2.fbnet_builder as fbnet_builder


def _create_input(input_dims):
    assert isinstance(input_dims, (tuple, list))
    nchw = np.prod(input_dims)
    ret = (torch.arange(nchw, dtype=torch.float32) - (nchw / 2.0)) / (nchw)
    ret = ret.reshape(*input_dims)
    return ret


class TestFBNetV2BlocksFactory(unittest.TestCase):
    OP_CFGS_DEFAULT = {
        "in_channels": 4,
        "out_channels": 4,
        "stride": 2,
        "_inputs_": [1, 4, 4, 4],
        "_gt_shape_": [1, 4, 2, 2],
        "bias": True,
    }
    OP_CFGS = {
        "default": OP_CFGS_DEFAULT,
        "conv_cfg": {
            **OP_CFGS_DEFAULT,
            "kernel_size": 3,
            "padding": 1,
            "bias": False,
        },
        "irf_cfg": {**OP_CFGS_DEFAULT, "expansion": 4, "bias": False},
        "irf_cfg_sefc": {
            **OP_CFGS_DEFAULT,
            "expansion": 4,
            "bias": False,
            "se_args": "se_fc",
            "width_divisor": 1,
        },
        "irf_cfg_seconvhsig": {
            **OP_CFGS_DEFAULT,
            "expansion": 4,
            "bias": False,
            "width_divisor": 1,
        },
    }

    # fmt: off
    # key: (op_name, cfg_name), default cfg will be used if only op_name is provided
    TEST_OP_EXPECTED_OUTPUT = {
        ("skip", "default"): ([1, 4, 2, 2], [0.91831, 0.881, 0.76907, 0.73176, 0.39075, 0.40302, 0.43984, 0.45211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # noqa
        ("conv_hs", "conv_cfg"): ([1, 4, 2, 2], [-0.15096, -0.1892, -0.17661, -0.18275, -0.07184, -0.10703, -0.0634, -0.09789, 0.00342, 0.09698, -0.03241, -0.07256, -0.03632, -0.04269, -0.08453, -0.05676]),  # noqa
        ("ir_k3", "irf_cfg"): ([1, 4, 2, 2], [0.01408, 0.07465, -0.06185, 0.05435, 0.0878, 0.09416, 0.07732, 0.0464, 0.11598, 0.17861, 0.09616, 0.22626, 0.16809, 0.23102, 0.26872, 0.27185]),  # noqa
        ("ir_k5", "irf_cfg"): ([1, 4, 2, 2], [0.00059, 0.01012, 0.12793, 0.07516, 0.04054, 0.00832, -0.03221, -0.00779, 0.00193, 0.00041, -0.02473, -0.04499, -0.41635, -0.18052, -0.26376, -0.18981]),  # noqa
        ("ir_k3", "irf_cfg_sefc"): ([1, 4, 2, 2], [0.03449, 0.03763, -0.0264, -0.01943, 0.15323, 0.19681, 0.18142, 0.26796, -0.02263, -0.07074, -0.00743, -0.10626, -0.07885, -0.11029, -0.0243, -0.05389]),  # noqa
        ("ir_k3_sehsig", "irf_cfg_seconvhsig"): ([1, 4, 2, 2], [-0.17406, -0.24675, -0.22749, -0.38714, -0.12084, -0.14361, -0.12785, -0.16972, 0.01255, 0.01106, 0.11432, 0.09041, 0.04951, 0.08074, 0.06063, 0.12473]),  # noqa
        ("ir_k5_sehsig", "default"): ([1, 4, 2, 2], [-0.01736, -0.0139, -0.01012, -0.05196, 0.02194, 0.03361, 0.03642, 0.0707, -0.02973, -0.03315, -0.0269, -0.01883, -0.00826, -0.00196, -0.0171, 0.02342]),  # noqa
        ("ir_pool", "default"): ([1, 4, 1, 1], [0.0, 0.1414, 0.49571, 0.43462]),  # noqa
        ("ir_pool_hs", "default"): ([1, 4, 1, 1], [-0.12981, 0.09867, -0.08033, 0.0543]),  # noqa
    }
    # fmt: on

    def _get_op_cfgs(self, op_name, op_cfg_name):
        assert (
            op_cfg_name in self.OP_CFGS
        ), f"op cfg name {op_cfg_name} not existed."
        op_cfg = self.OP_CFGS[op_cfg_name]
        op_cfg = copy.deepcopy(op_cfg)
        input_dims = op_cfg.pop("_inputs_")
        gt_shape = op_cfg.pop("_gt_shape_", None)

        output = self.TEST_OP_EXPECTED_OUTPUT.get((op_name, op_cfg_name), None)
        if output is None:
            assert op_cfg_name == "default"
            output = self.TEST_OP_EXPECTED_OUTPUT.get(op_name, None)
        if output is not None:
            gt_shape, gt_value = output
            gt_value = torch.FloatTensor(gt_value).reshape(gt_shape)
        else:
            gt_value = None

        if gt_shape is not None:
            gt_shape = torch.Size(gt_shape)

        return op_cfg, input_dims, gt_shape, gt_value

    def _test_primitive_check_output(self, device, op_name, op_cfg_name):
        torch.manual_seed(0)

        op_args, op_input_dims, gt_shape, gt_value = self._get_op_cfgs(
            op_name, op_cfg_name
        )
        op_func = fbnet_builder.PRIMITIVES.get(op_name)
        op = op_func(**op_args).to(device)
        op.eval()
        input = _create_input(op_input_dims).to(device)
        output = op(input)
        output = output.detach()

        self.assertEqual(output.shape, gt_shape)

        def _get_computed_output(result):
            ret = (
                f'("{op_name}", "{op_cfg_name}"): ({list(output.shape)}, '
                f"{[float('%.5f' % o) for o in output.view(-1).tolist()]})"
                ",  # noqa"
            )
            return ret

        if gt_value is not None:
            np.testing.assert_allclose(
                output,
                gt_value,
                rtol=0,
                atol=1e-4,
                err_msg=_get_computed_output(output),
            )
        else:
            print(
                f"Ground truth output for op {op_name} and cfg {op_cfg_name} "
                f"not provided. Computed output: \n{_get_computed_output(output)}"
            )

    def test_primitives_check_output(self):
        """ Make sures the primitives produce expected results """
        op_names = list(self.TEST_OP_EXPECTED_OUTPUT.keys())
        op_names = {
            (x, "default") if isinstance(x, str) else x for x in op_names
        }

        for op_name_info in op_names:
            op_name, op_cfg_name = op_name_info
            with self.subTest(op=op_name, cfg_name=op_cfg_name):
                print(f"Testing {op_name} with config {op_cfg_name}")
                self._test_primitive_check_output("cpu", op_name, op_cfg_name)

    def test_primitives_check_shape(self):
        """ Make sures the primitives runs """
        op_names = list(fbnet_builder.PRIMITIVES.get_names())
        op_names = {(x, "default") for x in op_names}

        for op_name_info in op_names:
            op_name, op_cfg_name = op_name_info
            with self.subTest(op=op_name, cfg_name=op_cfg_name):
                print(f"Testing {op_name} with config {op_cfg_name}")
                self._test_primitive_check_output("cpu", op_name, op_cfg_name)


if __name__ == "__main__":
    unittest.main()
