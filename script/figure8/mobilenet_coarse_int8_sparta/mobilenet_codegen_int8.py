from tkinter import N
import torch
import json
from sparta.artifact_specialization import generate_code

config = {}

id_shapes_name = "mobilenet_coarse_onnx/mobilenet_coarse_shape.json"
f = open(id_shapes_name)
id_shapes = json.load(f)

for name, content in id_shapes.items():
    if content['type'] != str(torch.nn.Conv2d):
        continue
    weight_shape = content['weight_shape'][0]
    conv_type = "conv1x1"
    if weight_shape[3] == 1:
        conv_type = "conv1x1"
        input_shape = content['in_shape'][0]
        output_shape = content['out_shape'][0]
        m = input_shape[0] * input_shape[2] * input_shape[3]
        k = input_shape[1]
        n = output_shape[1]
        config[name] = {'op_type': conv_type}
        config[name]['m'] = m
        config[name]['k'] = k
        config[name]['n'] = n
    elif weight_shape[1] == 1:
        conv_type = "depth_conv"
        input_shape = content['in_shape'][0]
        output_shape = content['out_shape'][0]
        weight_shape = content['weight_shape'][0]

        config[name] = {'op_type': conv_type}
        config[name]['channel'] = weight_shape[0]
        config[name]['in_height'] = input_shape[2]
        config[name]['in_width'] = input_shape[3]
        config[name]['out_height'] = output_shape[2]
        config[name]['out_width'] = output_shape[3]
        config[name]['batch_size'] = input_shape[0]
        config[name]['kernel_h'] = weight_shape[2]
        config[name]['kernel_w'] = weight_shape[3]
        config[name]['stride_h'] = content['stride'][0]
        config[name]['stride_w'] = content['stride'][1]
        config[name]['pad_h'] = content['padding'][0]
        config[name]['pad_w'] = content['padding'][1]
    else:
        print("skip conv3x3")
        continue

pattern = "mobilenet_coarse_int8"

result = generate_code(config, pattern)

with open("kernel_dict.json", "w") as outfile:
    json.dump(result, outfile)