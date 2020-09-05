#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
FBNet model builder

## Model Architecture Definition

We define a model architecture as a dict like the following:

```python
e6 = {"expansion": 6}
no_bias = {"bias": False}
backbone = [
    # [op, c, s, n, ...]
    # stage 0
    [("conv_k3", 32, 2, 1)],
    # stage 1
    [
        ("ir_k3", 64, 2, 2, e6, no_bias),
        ("ir_k5", 96, 1, 1, e6)
    ],
]
```
Here `backbone` is the defined architecture with two `stages`, and each stage
has one or more `building blocks`.

**Building blocks**

A building block `block` is represented as a tuple with four or more elements:
```python
    # [op, c, s, n, ...]
    block = ("ir_k3", 64, 2, 2, e6, no_bias)
```
where
  * `op` is the name of the block
  * `c` is the block output channel size,
  * `s` is the stride of the block,
  * `n` represents the number of repeats for this block.

Note that when `n` > 1, the stride `s` of the repeated blocks will be set to 1.
For example, blocks `[("ir_k5", 96, 2, 2, e6)]` is equivalent to
`[("ir_k5", 96, *2*, 1, e6), ("ir_k5", 96, *1*, 1, e6)]`.

Any additional arguments represent as dicts of argument pairs after `n` (like
`e6`, `no_bias` etc.) will be merged together in the order of appearance and pass
to the op's constructor.

All the supported building blocks are defined in
[fbnet_v2/blocks_factory.py](fbnet_v2/blocks_factory.py) and additional blocks
could be registered dynamically.

**Model architecture**

A list of building blocks represents a stage of the network
```python
    stage1 = [
        # block 0
        ("ir_k3", 64, 2, 2, e6, no_bias),
        # block 1
        ("ir_k5", 96, 1, 1, e6),
        ...
    ]
```
and a list of stages represent the architecture:
```python
    backbone = [
        # stage 0
        [("conv_k3", 32, 2, 1)],
        # stage 1
        [
            ("ir_k3", 64, 2, 2, e6, no_bias),
            ("ir_k5", 96, 1, 1, e6)
        ],
        ...
    ]
```
Note that the grouping of blocks to stages is only for convenience and does not
provide additional information to the architecture definition. We usually group
all the blocks that apply on the same spatial resolution feature map into the
same stage.

## Model Builder

We provide the following functions/classes to parse the above definition:

```python
from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder

e6 = {"expansion": 6}
bn_args = {"bn_args": {"name": "bn", "momentum": 0.003}}
arch_def = {
    # global arguments that will be applied to every op in the arch
    basic_args = {
        "relu_args": "swish",
    },
    "blocks": [
        # [op, c, s, n, ...]
        # stage 0
        [
            ("conv_k3", 4, 2, 1, bn_args)
        ],
        # stage 1
        [
            ("ir_k3", 8, 2, 2, e6, bn_args),
            ("ir_k5_sehsig", 8, 1, 1, e6, bn_args)
        ],
    ],
}
# unify architecture definition
arch_def = mbuilder.unify_arch_def(arch_def, ["blocks"])
# create builder
builder = mbuilder.FBNetBuilder(1.0)
# add global arguments
builder.add_basic_args(basic_args)
# build `nn.Module` from `blocks`
model = builder.build_blocks(arch_def["blocks"], dim_in=3)
# evaluation mode
model.eval()
```
Here
* `mbuilder.unify_arch_def` converts the arch definition to a way easier to
   handle later (convert to list of blocks, expand the repeats etc.). Only the
   dicts specified in the second argument will be unified and the rest will be
   unchanged.
* `builder.build_blocks(unified_arch_def["blocks"])` to create a `nn.Module`
   that corresponds to the architecture defined in `blocks`.
* We support specifying global default arguments to the builder that will be
  later override by each op by using `FBNetBuilder.add_basic_args(basic_args)`.
  Some common global arguments like the batch norm type and width divisor could
  be passed from the `FBNetBuilder` constructor as well.

"""

import copy
import logging
from collections import OrderedDict

import torch.nn as nn

import mobile_cv.arch.utils.helper as hp

from .blocks_factory import PRIMITIVES

logger = logging.getLogger(__name__)


class WrapperOp(nn.Module):
    def __init__(self, block_op, in_channels, out_channels, kwargs):
        super().__init__()
        self.block_op = block_op
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.op = PRIMITIVES.get(block_op)(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return self.op(x)


def parse_block_cfg(block_op, out_channels, stride=1, repeat=1, *args):
    assert all(isinstance(x, dict) for x in args), f"{args}"
    cfg = {"out_channels": out_channels, "stride": stride}
    [hp.update_dict(cfg, x) for x in args]

    ret = {"block_op": block_op, "block_cfg": cfg, "repeat": repeat}

    return ret


def parse_block_cfgs(block_cfgs):
    """ Parse block_cfgs like
            [
                [
                    ("ir_k3", 32, 2, 1)
                ],
                [
                    (
                        "ir_k3", 32, 2, 2,
                        {"expansion": 6, "dw_skip_bnrelu": True},
                        {"width_divisor": 8}
                    ),
                    ["conv_k1", 16, 1, 1]
                ],
            ]
        to:
            [
                [
                    {
                        "block_op": "ir_k3",
                        "block_cfg": {"out_channels": 32, "stride": 2}
                        "repeat: 1,
                    }
                ],
                [
                    {
                        "block_op": "ir_k3",
                        "block_cfg": {
                            "out_channels": 32, "stride": 2,
                            "expansion": 6, "dw_skip_bnrelu": True,
                            "width_divisor": 8
                        },
                        "repeat": 2,
                    },
                    {
                        "block_op": "conv_k1",
                        "block_cfg": {"out_channels": 16, "stride": 1},
                        "repeat": 1,
                    },
                ]
            ]
        The optional cfgs in each block (dicts) will be merged together in the
          order they appear in the dict.
    """
    assert isinstance(block_cfgs, list)
    ret = []
    for stage_cfg in block_cfgs:
        cur_stage = []
        for block_cfg in stage_cfg:
            assert isinstance(block_cfg, (list, tuple))
            cur_block = parse_block_cfg(*block_cfg)
            cur_stage.append(cur_block)
        ret.append(cur_stage)
    return ret


def _check_is_list(obj):
    assert isinstance(obj, (tuple, list)), f"{obj} is not a list"


def _check_lists_equal_size(*args):
    if len(args) == 0:
        return
    [_check_is_list(x) for x in args]
    size = len(args[0])
    assert all(len(x) == size for x in args), f"{args}"


def expand_repeats(blocks_info):
    """ Expand repeats in block cfg to multiple blocks and remove `_repeat_`
        Special handling for stride when repeat > 1 that the additionally expanded
            blocks will have stride 1
    """
    _check_is_list(blocks_info)
    ret = []
    for stage_cfgs in blocks_info:
        _check_is_list(stage_cfgs)
        cur_stage = []
        for block_cfg in stage_cfgs:
            assert isinstance(block_cfg, dict) and "block_cfg" in block_cfg
            cur_cfg = copy.deepcopy(block_cfg)
            repeat = cur_cfg.pop("repeat", 1)
            assert repeat >= 0
            # skip the block if repeat == 0
            if repeat == 0:
                continue
            expanded_cfgs = [copy.deepcopy(cur_cfg) for _ in range(repeat)]
            stride = cur_cfg["block_cfg"].get("stride", None)
            if repeat > 1 and stride is not None:
                # setup all strides to 1 except the first block
                for cur in expanded_cfgs[1:]:
                    cur["block_cfg"]["stride"] = 1
            cur_stage += expanded_cfgs
        ret.append(cur_stage)
    return ret


def flatten_stages(blocks_info):
    """ Flatten the blocks info from a list of list to a list
        Add 'stage_idx' and 'block_idx' to the blocks
    """
    _check_is_list(blocks_info)
    ret = []
    for stage_idx, stage_cfgs in enumerate(blocks_info):
        for block_idx, block_cfg in enumerate(stage_cfgs):
            cur = copy.deepcopy(block_cfg)
            cur["stage_idx"] = stage_idx
            cur["block_idx"] = block_idx
            ret.append(cur)
    return ret


def unify_arch_def_blocks(arch_def_blocks):
    """ unify an arch_def list
        [
            # [op, c, s, n, ...]
            # stage 0
            [("conv_k3", 32, 2, 1)],
            # stage 1
            [("ir_k3", 16, 1, 1, e1)],
        ]
        to
        [
            {
                "stage_idx": idx,
                "block_idx": idx,
                "block_cfg": {"out_channels": 32, "stride": 1, ...},
                "block_op": "conv_k3",
            },
            {}, ...
        ]
    """
    assert isinstance(arch_def_blocks, list)

    blocks_info = parse_block_cfgs(arch_def_blocks)
    blocks_info = expand_repeats(blocks_info)
    blocks_info = flatten_stages(blocks_info)

    return blocks_info


def unify_arch_def(arch_def, unify_names):
    """ unify an arch_def list
        {
            "blocks": [
                # [op, c, s, n, ...]
                # stage 0
                [("conv_k3", 32, 2, 1)],
                # stage 1
                [("ir_k3", 16, 1, 1, e1)],
            ]
        }
        to
        [
            "blocks": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    "block_cfg": {"out_channels": 32, "stride": 1, ...},
                    "block_op": "conv_k3",
                },
                {}, ...
            ],
        ]
    """
    assert isinstance(arch_def, dict)
    assert isinstance(unify_names, list)

    ret = copy.deepcopy(arch_def)
    for name in unify_names:
        if name not in ret:
            continue
        ret[name] = unify_arch_def_blocks(ret[name])

    return ret


def get_num_stages(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    ret = 0
    for x in arch_def_blocks:
        ret = max(x["stage_idx"], ret)
    ret = ret + 1
    return ret


def get_stages_dim_out(arch_def_blocks):
    """ Calculates the output channels of stage_idx

    Assuming the blocks in a stage are ordered, returns the c of tcns in the
    last block of the stage by going through all blocks in arch def
    Inputs: (dict) architecutre definition
            (int) stage idx
    Return: (list of int) stage output channels
    """
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    dim_out = [0] * get_num_stages(arch_def_blocks)
    for block in arch_def_blocks:
        stage_idx = block["stage_idx"]
        dim_out[stage_idx] = block["block_cfg"]["out_channels"]
    return dim_out


def get_num_blocks_in_stage(arch_def_blocks):
    """ Calculates the number of blocks in stage_idx

    Iterates over arch_def and counts the number of blocks
    Inputs: (dict) architecture definition
            (int) stage_idx
    Return: (list of int) number of blocks for each stage
    """
    assert isinstance(arch_def_blocks, list)
    assert all("stage_idx" in x for x in arch_def_blocks)
    nblocks = [0] * get_num_stages(arch_def_blocks)
    for block in arch_def_blocks:
        stage_idx = block["stage_idx"]
        nblocks[stage_idx] += 1
    return nblocks


def count_strides(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("block_cfg" in x for x in arch_def_blocks)
    ret = 1
    for stride in count_stride_each_block(arch_def_blocks):
        ret *= stride
    return ret


def count_stride_each_block(arch_def_blocks):
    assert isinstance(arch_def_blocks, list)
    assert all("block_cfg" in x for x in arch_def_blocks)
    ret = []
    for block in arch_def_blocks:
        stride = block["block_cfg"]["stride"]
        assert stride != 0, stride
        if stride > 0:
            ret.append(stride)
        else:
            ret.append(1.0 / -stride)
    return ret


BLOCK_KWARGS_NAME = "block_kwargs"


def add_block_kwargs(block, kwargs):
    if kwargs is None:
        return
    if BLOCK_KWARGS_NAME not in block:
        block[BLOCK_KWARGS_NAME] = {}
    block[BLOCK_KWARGS_NAME].update(kwargs)


def get_block_kwargs(block):
    return block.get(BLOCK_KWARGS_NAME, None)


def update_with_block_kwargs(dest, block):
    block_kwargs = get_block_kwargs(block)
    if block_kwargs is not None:
        assert isinstance(block_kwargs, dict)
        dest.update(block_kwargs)
    return dest


class FBNetBuilder(object):
    def __init__(self, width_ratio=1.0, bn_args="bn", width_divisor=1):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.width_divisor = width_divisor
        # basic arguments that will be provided to all primitivies, they could be
        #   overrided by primitive parameters
        self.basic_args = {
            "bn_args": hp.unify_args(bn_args),
            "width_divisor": width_divisor,
        }

    def add_basic_args(self, **kwargs):
        """ args that will be passed to all primitives, they could be
              overrided by primitive parameters
        """
        hp.update_dict(self.basic_args, kwargs)

    def build_blocks(
        self,
        blocks,
        stage_indices=None,
        dim_in=None,
        prefix_name="xif",
        **kwargs,
    ):
        """ blocks: [{}, {}, ...]

        Inputs: (list(int)) stages to add
                (list(int)) if block[0] is not connected to the most
                            recently added block, list specifies the input
                            dimensions of the blocks (as self.last_depth
                            will be inaccurate)
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        if stage_indices is not None:
            blocks = [x for x in blocks if x["stage_idx"] in stage_indices]

        if dim_in is not None:
            self.last_depth = dim_in
        assert (
            self.last_depth != -1
        ), "Invalid input dimension. Pass `dim_in` to `add_blocks`."

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            block_idx = block["block_idx"]
            block_op = block["block_op"]
            block_cfg = block["block_cfg"]
            cur_kwargs = update_with_block_kwargs(copy.deepcopy(kwargs), block)
            nnblock = self.build_block(
                block_op, block_cfg, dim_in=None, **cur_kwargs
            )
            nn_name = f"{prefix_name}{stage_idx}_{block_idx}"
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        ret.out_channels = self.last_depth
        return ret

    def build_block(self, block_op, block_cfg, dim_in=None, **kwargs):
        if dim_in is None:
            dim_in = self.last_depth
        assert "out_channels" in block_cfg
        block_cfg = copy.deepcopy(block_cfg)
        out_channels = block_cfg.pop("out_channels")
        out_channels = self._get_divisible_width(
            out_channels * self.width_ratio
        )
        # dicts appear later will override the configs in the earlier ones
        new_kwargs = hp.get_merged_dict(self.basic_args, block_cfg, kwargs)
        ret = WrapperOp(block_op, dim_in, out_channels, new_kwargs)
        self.last_depth = getattr(ret, "out_channels", out_channels)
        return ret

    def _get_divisible_width(self, width):
        ret = hp.get_divisible_by(
            int(width), self.width_divisor, self.width_divisor
        )
        return ret
