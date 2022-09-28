# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    
import math
import torch
from pytorchyolo import models

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch.utils import not_safe_to_prune
from nni.common.concrete_trace_utils import concrete_trace

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
dummy_input = torch.rand(16, 3, 320, 320)

def recompile_from_code(code: str, globals: dict):
  func_dict = {}
  globals = {
    **globals,
    'inf': math.inf,
    'nan': math.nan,
    'NoneType': type(None),
    'torch': torch,
    'device': torch.device,
  }
  exec(code, globals, func_dict)
  return func_dict['forward']

# run once before trace, for initiate parameters.
# otherwise some lazy-loading parameters may be initiated into proxies
out_a0 = model(dummy_input)
traced_model = concrete_trace(model, {'ims': dummy_input}, False)
recompiled = recompile_from_code(traced_model.code, model.forward.__globals__)

out_a1 = traced_model(dummy_input)
out_a2 = recompiled(model, dummy_input)
print('traced code:\n', traced_model.code)

print('out_a0 == out_a1:', torch.equal(out_a0, out_a1))
print('out_a0 == out_a2:', torch.equal(out_a0, out_a2))

dummy_input = torch.rand(16, 3, 320, 320)
out_b0 = model(dummy_input)
out_b1 = traced_model(dummy_input)
out_b2 = recompiled(model, dummy_input)
print('out_b0 == out_b1:', torch.equal(out_b0, out_b1))
print('out_b0 == out_b2:', torch.equal(out_b0, out_b2))