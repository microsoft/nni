# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import os

import numpy as np

LUT_PATH = "lut"


search_space = {
    # multi-stage definition for candidate layers
    # here two stages are defined for PFLD searching
    "stages": {
        "stage_0": {
            "ops": [
                "mb_k3_res",
                "mb_k3_e2_res",
                "mb_k3_res_d3",
                "mb_k5_res",
                "mb_k5_e2_res",
                "sep_k3",
                "sep_k5",
                "gh_k3",
                "gh_k5",
            ],
            "layer_num": 2,
        },
        "stage_1": {
            "ops": [
                "mb_k3_e2_res",
                "mb_k3_e4_res",
                "mb_k3_e2_res_se",
                "mb_k3_res_d3",
                "mb_k5_res",
                "mb_k5_e2_res",
                "mb_k5_res_se",
                "mb_k5_e2_res_se",
                "gh_k5",
            ],
            "layer_num": 3,
        },
    },
    # necessary information of layers for NAS
    # the basic information is as (input_channels, height, width)
    "input_shape": [
        (32, 14, 14),
        (32, 14, 14),
        (32, 14, 14),
        (64, 7, 7),
        (64, 7, 7),
    ],
    # output channels for each layer
    "channel_size": [32, 32, 64, 64, 64],
    # stride for each layer
    "strides": [1, 1, 2, 1, 1],
    # height of feature map for each layer
    "fm_size": [14, 14, 7, 7, 7],
}


class NASConfig:
    def __init__(
        self,
        perf_metric="flops",
        lut_load=False,
        arch_search=True,
        model_dir=None,
        nas_lr=0.01,
        nas_weight_decay=5e-4,
        mode="mul",
        alpha=0.25,
        beta=0.8,
        start_epoch=50,
        init_temperature=5.0,
        exp_anneal_rate=np.exp(-0.045),
        search_space=None,
    ):
        # LUT of performance metric
        # flops means the multiplies, latency means the time cost on platform
        self.perf_metric = perf_metric
        assert perf_metric in [
            "flops",
            "latency",
        ], "perf_metric should be ['flops', 'latency']"
        # wether load or create lut file
        self.lut_load = lut_load
        self.arch_search = arch_search
        # necessary dirs
        self.lut_en = model_dir is not None
        if self.lut_en:
            self.model_dir = model_dir
            os.makedirs(model_dir, exist_ok=True)
            self.lut_path = os.path.join(model_dir, LUT_PATH)
            os.makedirs(self.lut_path, exist_ok=True)
        # NAS learning setting
        self.nas_lr = nas_lr
        self.nas_weight_decay = nas_weight_decay
        # hardware-aware loss setting
        self.mode = mode
        assert mode in ["mul", "add"], "mode should be ['mul', 'add']"
        self.alpha = alpha
        self.beta = beta
        # NAS training setting
        self.start_epoch = start_epoch
        self.init_temperature = init_temperature
        self.exp_anneal_rate = exp_anneal_rate
        # definition of search blocks and space
        self.search_space = search_space
