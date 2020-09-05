#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import e1, e6

BASIC_ARGS = {"relu_args": "swish"}

IRF_CFG = {"less_se_channels": True, "zero_last_bn_gamma": True}


MODEL_ARCH_EFFICIENT_NET = {
    "eff_0": {
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [["ir_k3_se", 16, 1, 1, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 24, 2, 2, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 40, 2, 2, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 80, 2, 3, e6, IRF_CFG],
                ["ir_k5_se", 112, 1, 3, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 192, 2, 4, e6, IRF_CFG],
                ["ir_k3_se", 320, 1, 1, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_1": {
        "input_size": 240,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [["ir_k3_se", 16, 1, 2, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 24, 2, 3, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 40, 2, 3, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 80, 2, 4, e6, IRF_CFG],
                ["ir_k5_se", 112, 1, 4, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 192, 2, 5, e6, IRF_CFG],
                ["ir_k3_se", 320, 1, 2, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 1280, 1, 1]],
        ],
    },
    "eff_2": {
        "input_size": 260,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 32, 2, 1]],
            # stage 1
            [["ir_k3_se", 16, 1, 2, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 24, 2, 3, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 48, 2, 3, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 88, 2, 4, e6, IRF_CFG],
                ["ir_k5_se", 120, 1, 4, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 208, 2, 5, e6, IRF_CFG],
                ["ir_k3_se", 352, 1, 2, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 1408, 1, 1]],
        ],
    },
    "eff_3": {
        "input_size": 300,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 40, 2, 1]],
            # stage 1
            [["ir_k3_se", 24, 1, 2, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 32, 2, 3, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 48, 2, 3, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 96, 2, 5, e6, IRF_CFG],
                ["ir_k5_se", 136, 1, 5, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 232, 2, 6, e6, IRF_CFG],
                ["ir_k3_se", 384, 1, 2, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 1536, 1, 1]],
        ],
    },
    "eff_4": {
        "input_size": 380,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 48, 2, 1]],
            # stage 1
            [["ir_k3_se", 24, 1, 2, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 32, 2, 4, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 56, 2, 4, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 112, 2, 6, e6, IRF_CFG],
                ["ir_k5_se", 160, 1, 6, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 272, 2, 8, e6, IRF_CFG],
                ["ir_k3_se", 448, 1, 2, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 1792, 1, 1]],
        ],
    },
    "eff_5": {
        # width: 1.6, depth: 2.2, res: 456
        "input_size": 456,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 56, 2, 1]],
            # stage 1
            [["ir_k3_se", 32, 1, 3, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 40, 2, 5, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 64, 2, 5, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 128, 2, 7, e6, IRF_CFG],
                ["ir_k5_se", 184, 1, 7, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 312, 2, 9, e6, IRF_CFG],
                ["ir_k3_se", 512, 1, 3, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 2048, 1, 1]],
        ],
    },
    "eff_6": {
        # width: 1.8, depth: 2.6, res: 528
        "input_size": 528,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 64, 2, 1]],
            # stage 1
            [["ir_k3_se", 32, 1, 3, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 48, 2, 6, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 72, 2, 6, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 144, 2, 8, e6, IRF_CFG],
                ["ir_k5_se", 208, 1, 8, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 352, 2, 11, e6, IRF_CFG],
                ["ir_k3_se", 576, 1, 3, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 2304, 1, 1]],
        ],
    },
    "eff_7": {
        # width: 2.0, depth: 3.1, res: 600
        "input_size": 600,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 72, 2, 1]],
            # stage 1
            [["ir_k3_se", 40, 1, 4, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 56, 2, 7, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 88, 2, 7, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 168, 2, 10, e6, IRF_CFG],
                ["ir_k5_se", 232, 1, 10, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 392, 2, 13, e6, IRF_CFG],
                ["ir_k3_se", 648, 1, 4, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 2568, 1, 1]],
        ],
    },
    "eff_8": {
        # width: 2.2, depth: 3.6, res: 672
        "input_size": 672,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3", 72, 2, 1]],
            # stage 1
            [["ir_k3_se", 40, 1, 4, e1, IRF_CFG]],
            # stage 2
            [["ir_k3_se", 56, 2, 7, e6, IRF_CFG]],
            # stage 3
            [["ir_k5_se", 96, 2, 7, e6, IRF_CFG]],
            # stage 4
            [
                ["ir_k3_se", 184, 2, 11, e6, IRF_CFG],
                ["ir_k5_se", 248, 1, 11, e6, IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k5_se", 424, 2, 14, e6, IRF_CFG],
                ["ir_k3_se", 712, 1, 4, e6, IRF_CFG],
            ],
            # stage 6
            [["conv_k1", 2824, 1, 1]],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_EFFICIENT_NET)
