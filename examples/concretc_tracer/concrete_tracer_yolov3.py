# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
    
import torch
from pytorchyolo import models

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch.utils import not_safe_to_prune
from nni.common.concrete_trace_utils import concrete_trace

# The Yolo can be downloaded at https://github.com/eriklindernoren/PyTorch-YOLOv3.git
prefix = 'C:\\Works\\PyTorch-YOLOv3' # replace this path with yours
# Load the YOLO model
model = models.load_model(
  "%s/config/yolov3.cfg" % prefix, 
  "%s/weights/yolov3.weights" % prefix).cpu()
model.eval()
dummy_input = torch.rand(8, 3, 320, 320)

def recompile_from_code(code: str, globals: dict):
  func_dict = {}
  exec(code, globals, func_dict)
  return func_dict['forward']

traced_model = concrete_trace(model, {'x': dummy_input}, False)

out0 = model(dummy_input)
out1 = traced_model(dummy_input)
out2 = recompile_from_code(traced_model.code, model.forward.__globals__)(model, dummy_input)
print('traced code:\n', traced_model.code)
print('out0 == out1:', torch.equal(out0, out1))
print('out0 == out2:', torch.equal(out0, out1))
