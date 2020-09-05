#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import mobile_cv.common.misc.registry as registry

from . import modeldef_utils as mdu
from .modeldef_utils import _ex, e1, e3, e4, e6

MODEL_ARCH = registry.Registry("cls_arch_factory")


MODEL_ARCH_DEFAULT = {
    "default": {
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
            # stage 2
            [("ir_k3", 24, 2, 2, e6)],
            # stage 3
            [("ir_k3", 32, 2, 3, e6)],
            # stage 4
            [("ir_k3", 64, 2, 4, e6), ("ir_k3", 96, 1, 3, e6)],
            # stage 5
            [("ir_k3", 160, 2, 3, e6), ("ir_k3", 320, 1, 1, e6)],
            # stage 6
            [("conv_k1", 1280, 1, 1)],
        ]
    },
    "mnv3": {
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [("conv_k3_hs", 16, 2, 1)],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [["ir_k3", 24, 2, 1, e4], ["ir_k3", 24, 1, 1, e3]],
            # stage 3
            [["ir_k5_sehsig", 40, 2, 3, e3]],
            # stage 4
            [
                ["ir_k3_hs", 80, 2, 1, e6],
                ["ir_k3_hs", 80, 1, 1, _ex(2.5)],
                ["ir_k3_hs", 80, 1, 2, _ex(2.3)],
                ["ir_k3_sehsig_hs", 112, 1, 2, e6],
            ],
            # stage 5
            [["ir_k5_sehsig_hs", 160, 2, 3, e6]],
            # stage 6
            [["ir_pool_hs", 1280, 1, 1, e6]],
        ]
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DEFAULT)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_DEFAULT))


MODEL_ARCH_FBNET = {
    "fbnet_a": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["skip", 16, 1, 1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e3],
                ["ir_k3", 24, 1, 1, e1],
                ["skip", 24, 1, 1],
                ["skip", 24, 1, 1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k3", 32, 1, 1, e3],
                ["ir_k5", 32, 1, 1, e1],
                ["ir_k3", 32, 1, 1, e3],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5_g2", 64, 1, 1, e1],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k3", 112, 1, 1, e6],
                ["ir_k5_g2", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e3],
                ["ir_k3_g2", 112, 1, 1, e1],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e3],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 352, 1, 1, e6],
            ],
            # stage 5
            [("conv_k1", 1504, 1, 1)],
        ],
    },
    "fbnet_b": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k5", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                ["ir_k3", 32, 1, 1, e6],
                # ["ir_k3_sep", 32, 1, 1, e6]],
                ["ir_k5", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["skip", 64, 1, 1],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k3", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e1],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e1],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
            # stage 5
            [("conv_k1", 1984, 1, 1)],
        ],
    },
    "fbnet_c": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["skip", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5", 32, 2, 1, e6],
                ["ir_k5", 32, 1, 1, e3],
                # ["ir_k3_sep", 32, 1, 1, e6],
                ["ir_k5", 32, 1, 1, e6],
                ["ir_k3", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5", 64, 2, 1, e6],
                ["ir_k5", 64, 1, 1, e3],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 64, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                #  ["ir_k3_sep", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e6],
                ["ir_k5", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5", 184, 2, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k5", 184, 1, 1, e6],
                ["ir_k3", 352, 1, 1, e6],
            ],
            # stage 6
            [("conv_k1", 1984, 1, 1)],
        ],
    },
    "fbnet_96": {
        "input_size": 96,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3", 8, 2, 1]],
            # stage 1
            [["ir_k3", 8, 1, 1, e1]],
            # stage 2
            [["ir_k3", 16, 2, 1, e6], ["ir_k3", 16, 1, 1, e6]],
            # stage 3
            [["ir_k5", 16, 2, 1, e6], ["ir_k5", 16, 1, 1, e6]],
            # stage 4
            [["ir_k5", 24, 2, 1, e6], ["ir_k3", 40, 1, 1, e6]],
            # stage 5
            [
                ["ir_k5", 72, 2, 1, e6],
                ["ir_k5", 72, 1, 1, e6],
                ["ir_k5", 128, 1, 1, e6],
            ],
            # stage 6
            [("conv_k1", 1416, 1, 1)],
        ],
        "preprocessing": "resNet",
    },
    "fbnet_ase": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["skip", 16, 1, 1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e3],
                ["ir_k3", 24, 1, 1, e1],
                ["skip", 24, 1, 1],
                ["skip", 24, 1, 1],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, e6],
                ["ir_k3_sehsig", 32, 1, 1, e3],
                ["ir_k5_sehsig", 32, 1, 1, e1],
                ["ir_k3_sehsig", 32, 1, 1, e3],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e3],
                ["ir_k5_hs", 64, 1, 1, e1],
                ["ir_k5_hs", 64, 1, 1, e6],
                ["ir_k3_hs", 112, 1, 1, e6],
                ["ir_k5_hs", 112, 1, 1, e1],
                ["ir_k5_sehsig_hs", 112, 1, 1, e3],
                ["ir_k3_sehsig_hs", 112, 1, 1, e1],
            ],
            # stage 5
            [
                ["ir_k5_sehsig_hs", 184, 2, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e3],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
            ],
            # stage 6
            [["ir_pool_hs", 1504, 1, 1, e6]],
        ],
    },
    "fbnet_bse": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 0
            [["ir_k3", 16, 1, 1, e1]],
            # stage 1
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k5", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 2
            [
                ["ir_k5_sehsig", 32, 2, 1, e6],
                ["ir_k5_sehsig", 32, 1, 1, e3],
                ["ir_k3_sehsig", 32, 1, 1, e6],
                ["ir_k5_sehsig", 32, 1, 1, e6],
            ],
            # stage 3
            [
                ["ir_k5_hs", 64, 2, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e1],
                ["skip", 64, 1, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e3],
                ["ir_k5_hs", 112, 1, 1, e6],
                ["ir_k3_sehsig_hs", 112, 1, 1, e1],
                ["ir_k5_sehsig_hs", 112, 1, 1, e1],
                ["ir_k5_sehsig_hs", 112, 1, 1, e3],
            ],
            # stage 4
            [
                ["ir_k5_sehsig_hs", 184, 2, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e1],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
            ],
            # stage 4
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "fbnet_cse": {
        "input_size": 224,
        "blocks": [
            # [op, c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["skip", 24, 1, 1],
                ["ir_k3", 24, 1, 1, e1],
                ["ir_k3", 24, 1, 1, e1],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, e6],
                ["ir_k5_sehsig", 32, 1, 1, e3],
                ["ir_k5_sehsig", 32, 1, 1, e6],
                ["ir_k3_sehsig", 32, 1, 1, e6],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e3],
                ["ir_k5_hs", 64, 1, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e6],
                ["ir_k5_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e3],
            ],
            # stage 5
            [
                ["ir_k5_sehsig_hs", 184, 2, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "fbnet_dse": {
        "input_size": 224,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 2, e1]],
            # stage 2
            [
                ["ir_k3", 24, 2, 1, e6],
                ["ir_k3", 24, 1, 1, e6],
                ["ir_k3", 24, 1, 1, e6],
                ["ir_k3", 24, 1, 1, e6],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, e6],
                ["ir_k5_sehsig", 32, 1, 1, e6],
                ["ir_k5_sehsig", 32, 1, 2, e6],
                ["ir_k3_sehsig", 32, 1, 2, e6],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e6],
                ["ir_k5_hs", 64, 1, 1, e6],
                ["ir_k5_hs", 96, 1, 1, e6],
                ["ir_k5_hs", 96, 1, 1, e6],
                ["ir_k5_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 112, 1, 1, e6],
                ["ir_k5_sehsig_hs", 128, 1, 1, e6],
            ],
            # stage 5
            [
                ["ir_k5_sehsig_hs", 184, 2, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k5_sehsig_hs", 184, 1, 1, e6],
                ["ir_k3_sehsig_hs", 256, 1, 1, e6],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_FBNET)
MODEL_ARCH.register_dict(mdu.get_i8f_models(MODEL_ARCH_FBNET))
