#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy


# add int8 friendly version of the model
def get_i8f_models(model_def):
    ret = {}
    for name, arch in model_def.items():
        new_name = name + "_i8f"
        new_arch = copy.deepcopy(arch)
        if "basic_args" not in new_arch:
            new_arch["basic_args"] = {}
        new_arch["basic_args"]["dw_skip_bnrelu"] = True
        ret[new_name] = new_arch
    return ret


def _ex(x, always_pw=None):
    ret = {"expansion": x}
    if always_pw is not None:
        ret["always_pw"] = always_pw
    return ret


e6 = _ex(6)
e4 = _ex(4)
e3 = _ex(3)
e2 = _ex(2)
e1 = _ex(1)
e1p = _ex(1, always_pw=True)
