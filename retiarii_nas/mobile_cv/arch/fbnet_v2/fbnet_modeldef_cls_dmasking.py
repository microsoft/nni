#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e6

BASIC_ARGS = {}

IRF_CFG = {"less_se_channels": False}

MODEL_ARCH_DMASKING_NET = {
    "dmasking_f1": {
        # nparams: 5.998952, nflops 55.747008
        "input_size": 128,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 8, 2, 1]],
            # stage 1
            [["ir_k5", 8, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(4.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 24, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 56, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(4.1212), IRF_CFG],
                ["ir_k3_sehsig_hs", 56, 1, 1, _ex(5.1246), IRF_CFG],
                ["skip", 80, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(4.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 80, 1, 1, _ex(1.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 144, 2, 1, _ex(4.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(5.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 144, 1, 1, _ex(6.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1600, 1, 1, e6]],
        ],
    },
    "dmasking_f4": {
        # nparams: 6.993656, nflops 234.689136
        "input_size": 224,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],
            # stage 1
            [["ir_k3", 16, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 32, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 104, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 104, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "dmasking_l2_hs": {
        # nparams: 8.49 nflops: 422.04
        "input_size": 256,
        "basic_args": BASIC_ARGS,
        "blocks": [
            [["conv_k3_hs", 16, 2, 1]],
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],
            [
                ["ir_k5_hs", 24, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5_hs", 24, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            [
                ["ir_k5_sehsig", 40, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 32, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            [
                ["ir_k5_hs", 64, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 112, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 112, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            [
                ["ir_k3_sehsig_hs", 184, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 184, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 224, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
    "dmasking_l3": {
        # nparams: 9.402096, nflops 750.681952
        "input_size": 288,
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 24, 2, 1]],
            # stage 1
            [["ir_k3", 24, 1, 1, e1, IRF_CFG]],
            # stage 2
            [
                ["ir_k5", 32, 2, 1, _ex(5.4566), IRF_CFG],
                ["ir_k5", 32, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k3", 32, 1, 1, _ex(1.7912), IRF_CFG],
                ["ir_k5", 32, 1, 1, _ex(1.7912), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k5_sehsig", 48, 2, 1, _ex(5.3501), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(3.5379), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(4.5379), IRF_CFG],
                ["ir_k5_sehsig", 40, 1, 1, _ex(4.5379), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k5_hs", 72, 2, 1, _ex(5.7133), IRF_CFG],
                ["ir_k3_hs", 72, 1, 1, _ex(2.1212), IRF_CFG],
                ["skip", 72, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 72, 1, 1, _ex(3.1246), IRF_CFG],
                ["ir_k3_hs", 120, 1, 1, _ex(5.0333), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(2.5070), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(1.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(2.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
                ["ir_k5_sehsig_hs", 120, 1, 1, _ex(3.7712), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_sehsig_hs", 192, 2, 1, _ex(5.5685), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(2.8400), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(4.8754), IRF_CFG],
                ["ir_k5_sehsig_hs", 192, 1, 1, _ex(4.8754), IRF_CFG],
                ["skip", 240, 1, 1, _ex(6.5245), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET)
