# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import torch

from nni.common.concrete_trace_utils import concrete_trace

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
dummy_input = torch.rand(16, 3, 320, 320)

def recompile_from_code(code: str, global_vars: dict):
    func_dict = {}
    global_vars = {
        **global_vars,
        'inf': math.inf,
        'nan': math.nan,
        'NoneType': type(None),
        'torch': torch,
        'device': torch.device,
    }
    exec(code, global_vars, func_dict)
    return func_dict['forward']

# run once before trace, for initiate parameters.
# otherwise some lazy-loading parameters may be initiated into proxies
out_a0 = model(dummy_input)
traced_model = concrete_trace(model, {'ims': dummy_input})
recompiled = recompile_from_code(traced_model.code, model.forward.__globals__)
print('traced code:\n', traced_model.code)

out_a1 = traced_model(dummy_input)
print('is the output of traced_model equals with orig_model using tracing input:', torch.equal(out_a0, out_a1))

dummy_input = torch.rand_like(dummy_input)
out_b0 = model(dummy_input)
out_b1 = traced_model(dummy_input)
print('is the output of traced_model equals with orig_model using another random input:', torch.equal(out_b0, out_b1))
