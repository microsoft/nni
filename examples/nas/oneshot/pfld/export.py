# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import onnx
import onnxsim
import os
import torch

from lib.builder import search_space
from lib.ops import PRIMITIVES
from nni.algorithms.nas.pytorch.fbnet import (
    LookUpTable,
    NASConfig,
    model_init,
)


parser = argparse.ArgumentParser(description="Export the ONNX model")
parser.add_argument("--net", default="subnet", type=str)
parser.add_argument("--supernet", default="", type=str, metavar="PATH")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("--num_points", default=106, type=int)
parser.add_argument("--img_size", default=112, type=int)
parser.add_argument("--onnx", default="./output/pfld.onnx", type=str)
parser.add_argument("--onnx_sim", default="./output/subnet.onnx", type=str)
args = parser.parse_args()

os.makedirs("./output", exist_ok=True)

if args.net == "subnet":
    from lib.subnet import PFLDInference
else:
    raise ValueError("Network is not implemented")

check = torch.load(args.supernet, map_location=torch.device("cpu"))
sampled_arch = check["arch_sample"]

nas_config = NASConfig(search_space=search_space)
lookup_table = LookUpTable(config=nas_config, primitives=PRIMITIVES)
pfld_backbone = PFLDInference(lookup_table, sampled_arch, args.num_points)

pfld_backbone.eval()
check_sub = torch.load(args.resume, map_location=torch.device("cpu"))
param_dict = check_sub["pfld_backbone"]
model_init(pfld_backbone, param_dict)

print("Convert PyTorch model to ONNX.")
dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(
    pfld_backbone,
    dummy_input,
    args.onnx,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
)

print("Check ONNX model.")
model = onnx.load(args.onnx)

print("Simplifying the ONNX model.")
model_opt, check = onnxsim.simplify(args.onnx)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_opt, args.onnx_sim)
print("Onnx model simplify Ok!")
