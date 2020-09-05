#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from . import lut_schema


def compute_flops(lut_ops, unit=1e6):
    nparams, nflops = 0.0, 0.0
    for op_info in lut_ops:
        assert isinstance(op_info, (lut_schema.OpInfo, lut_schema.LutItem))
        cnp = op_info.op.get_nparams()
        cflops = op_info.op.get_flops(op_info.input_shapes)
        nparams += cnp
        nflops += cflops

    return nparams / unit, nflops / unit
