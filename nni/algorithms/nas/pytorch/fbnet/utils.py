# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import gc  # noqa: F401
import os
import timeit
import torch

import numpy as np
import torch.nn as nn

from nni.compression.pytorch.utils.counter import count_flops_params

LUT_FILE = "lut.npy"
LUT_PATH = "lut"


class NASConfig:
    def __init__(
        self,
        perf_metric="flops",
        lut_load=False,
        model_dir=None,
        nas_lr=0.01,
        nas_weight_decay=5e-4,
        mode="mul",
        alpha=0.25,
        beta=0.6,
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


class RegularizerLoss(nn.Module):
    """Auxilliary loss for hardware-aware NAS."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : class
            to manage the configuration for NAS training, and search space etc.
        """
        super(RegularizerLoss, self).__init__()
        self.mode = config.mode
        self.alpha = config.alpha
        self.beta = config.beta

    def forward(self, perf_cost, batch_size=1):
        """
        Parameters
        ----------
        perf_cost : tensor
            the accumulated performance cost
        batch_size : int
            batch size for normalization

        Returns
        -------
        output: tensor
            the hardware-aware constraint loss
        """
        if self.mode == "mul":
            log_loss = torch.log(perf_cost / batch_size) ** self.beta
            return self.alpha * log_loss
        elif self.mode == "add":
            linear_loss = (perf_cost / batch_size) ** self.beta
            return self.alpha * linear_loss
        else:
            raise NotImplementedError


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k

    Parameters
    ----------
    output : pytorch tensor
        output, e.g., predicted value
    target : pytorch tensor
        label
    topk : tuple
        specify top1 and top5

    Returns
    -------
    list
        accuracy of top1 and top5
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


def model_init(model, state_dict, replace=[]):
    """Initialize the model from state_dict."""
    prefix = "module."
    param_dict = dict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[7:]
        param_dict[k] = v

    for k, (name, m) in enumerate(model.named_modules()):
        if replace:
            for layer_replace in replace:
                assert len(layer_replace) == 3, "The elements should be three."
                pre_scope, key, replace_key = layer_replace
                if pre_scope in name:
                    name = name.replace(key, replace_key)

        # Copy the state_dict to current model
        if (name + ".weight" in param_dict) or (
            name + ".running_mean" in param_dict
        ):
            if isinstance(m, nn.BatchNorm2d):
                shape = m.running_mean.shape
                if shape == param_dict[name + ".running_mean"].shape:
                    if m.weight is not None:
                        m.weight.data = param_dict[name + ".weight"]
                        m.bias.data = param_dict[name + ".bias"]
                    m.running_mean = param_dict[name + ".running_mean"]
                    m.running_var = param_dict[name + ".running_var"]

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                shape = m.weight.data.shape
                if shape == param_dict[name + ".weight"].shape:
                    m.weight.data = param_dict[name + ".weight"]
                    if m.bias is not None:
                        m.bias.data = param_dict[name + ".bias"]

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data = param_dict[name + ".weight"]
                if m.bias is not None:
                    m.bias.data = param_dict[name + ".bias"]


class LookUpTable:
    """Build look-up table for NAS."""

    def __init__(self, config, primitives):
        """
        Parameters
        ----------
        config : class
            to manage the configuration for NAS training, and search space etc.
        """
        self.config = config
        # definition of search blocks and space
        self.search_space = config.search_space
        # layers for NAS
        self.cnt_layers = len(self.search_space["input_shape"])
        # constructors for each operation
        self.lut_ops = {
            stage_name: {
                op_name: primitives[op_name]
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

                    # measured in Flops
                    in_shape = self.layer_in_shapes[layer_id]
                    x = (1, in_shape[0], in_shape[1], in_shape[2])
                    flops, _, _ = count_flops_params(op, x, verbose=False)
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
