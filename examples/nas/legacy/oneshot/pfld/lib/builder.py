# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function


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
