# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import gc  # noqa: F401
import os
import timeit
import torch

import numpy as np

from lib.ops import PRIMITIVES
from lib.utils import count_flops, model_init

LUT_FILE = "lut.npy"


def supernet_sample(model, state_dict, sampled_arch=[], lookup_table=None):
    """
    Initialize the searched sub-model from supernet.

    Parameters
    ----------
    model : pytorch model
        the created subnet
    state_dict : checkpoint
        the checkpoint of supernet, including the pre-trained params
    sampled_arch : list of str
        the searched layer names of the subnet
    lookup_table : class
        to manage the candidate ops, layer information and layer performance
    """
    replace = list()
    stages = [stage for stage in lookup_table.layer_num]
    stage_lnum = [lookup_table.layer_num[stage] for stage in stages]

    if sampled_arch:
        layer_id = 0
        for i, stage in enumerate(stages):
            ops_names = [op_name for op_name in lookup_table.lut_ops[stage]]
            for j in range(stage_lnum[i]):
                searched_op = sampled_arch[layer_id]
                op_i = ops_names.index(searched_op)
                replace.append(
                    [
                        "blocks.{}.".format(layer_id),
                        "blocks.{}.op.".format(layer_id),
                        "blocks.{}.{}.".format(layer_id, op_i),
                    ]
                )
                layer_id += 1
    model_init(model, state_dict, replace=replace)


class LookUpTable:
    """Build look-up table for NAS."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : class
            to manage the configuration for NAS training, and search space etc.
        """
        # definition of search blocks and space
        self.search_space = config.search_space
        # layers for NAS
        self.cnt_layers = len(self.search_space["input_shape"])
        # constructors for each operation
        self.lut_ops = {
            stage_name: {
                op_name: PRIMITIVES[op_name]
                for op_name in self.search_space["stages"][stage_name]["ops"]
            }
            for stage_name in self.search_space["stages"]
        }
        self.layer_num = {
            stage_name: self.search_space["stages"][stage_name]["layer_num"]
            for stage_name in self.search_space["stages"]
        }

        # arguments for the ops constructors, input_shapes just for convinience
        self.layer_configs, self.layer_in_shapes = self._layer_configs()

        # lookup_table
        self.perf_metric = config.perf_metric

        if config.lut_en:
            self.lut_perf = None
            self.lut_file = os.path.join(config.lut_path, LUT_FILE)
            if config.lut_load:
                self._load_from_file()
            else:
                self._create_perfs()

    def _layer_configs(self):
        """Generate basic params for different layers."""
        # layer_configs are : c_in, c_out, stride, fm_size
        layer_configs = [
            [
                self.search_space["input_shape"][layer_id][0],
                self.search_space["channel_size"][layer_id],
                self.search_space["strides"][layer_id],
                self.search_space["fm_size"][layer_id],
            ]
            for layer_id in range(self.cnt_layers)
        ]

        # layer_in_shapes are (C_in, input_w, input_h)
        layer_in_shapes = self.search_space["input_shape"]

        return layer_configs, layer_in_shapes

    def _create_perfs(self, cnt_of_runs=200):
        """Create performance cost for each op."""
        if self.perf_metric == "latency":
            self.lut_perf = self._calculate_latency(cnt_of_runs)
        elif self.perf_metric == "flops":
            self.lut_perf = self._calculate_flops()

        self._write_lut_to_file()

    def _calculate_flops(self, eps=0.001):
        """FLOPs cost."""
        flops_lut = [{} for i in range(self.cnt_layers)]
        layer_id = 0

        for stage_name in self.lut_ops:
            stage_ops = self.lut_ops[stage_name]
            ops_num = self.layer_num[stage_name]

            for _ in range(ops_num):
                for op_name in stage_ops:
                    layer_config = self.layer_configs[layer_id]
                    key_params = {"fm_size": layer_config[3]}
                    op = stage_ops[op_name](*layer_config[0:3], **key_params)

                    # measured in micro-second
                    flops = count_flops(op, self.layer_in_shapes[layer_id])
                    flops = eps if flops == 0.0 else flops
                    flops_lut[layer_id][op_name] = float(flops)
                layer_id += 1

        return flops_lut

    def _calculate_latency(self, cnt_of_runs):
        """Latency cost."""
        LATENCY_BATCH_SIZE = 1
        latency_lut = [{} for i in range(self.cnt_layers)]
        layer_id = 0

        for stage_name in self.lut_ops:
            stage_ops = self.lut_ops[stage_name]
            ops_num = self.layer_num[stage_name]

            for _ in range(ops_num):
                for op_name in stage_ops:
                    layer_config = self.layer_configs[layer_id]
                    key_params = {"fm_size": layer_config[3]}
                    op = stage_ops[op_name](*layer_config[0:3], **key_params)
                    input_data = torch.randn(
                        (LATENCY_BATCH_SIZE, *self.layer_in_shapes[layer_id])
                    )
                    globals()["op"], globals()["input_data"] = op, input_data
                    total_time = timeit.timeit(
                        "output = op(input_data)",
                        setup="gc.enable()",
                        globals=globals(),
                        number=cnt_of_runs,
                    )
                    # measured in micro-second
                    latency_lut[layer_id][op_name] = (
                        total_time / cnt_of_runs / LATENCY_BATCH_SIZE * 1e6
                    )
                layer_id += 1

        return latency_lut

    def _write_lut_to_file(self):
        """Save lut as numpy file."""
        np.save(self.lut_file, self.lut_perf)

    def _load_from_file(self):
        """Load numpy file."""
        self.lut_perf = np.load(self.lut_file, allow_pickle=True)
