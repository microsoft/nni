from tkinter import N
import torch
import json
from sparta.artifact_specialization import generate_code

tesaid_2_names_file = "artifact_hubert_coarse_onnx_with_tesa/tesaid_2_names"
tesaid_2_names = torch.load(tesaid_2_names_file)
config = {}

id_shapes_name = "shape.json"
f = open(id_shapes_name)
id_shapes = json.load(f)

for tesa_id, name_list in tesaid_2_names.items():
    pytorch_name, onnx_name = name_list[0], name_list[1]
    shape_dict = id_shapes[pytorch_name]
    if shape_dict['type'] == str(torch.nn.Conv1d):
        continue
    # import ipdb; ipdb.set_trace()
    if len(shape_dict['in_shape'][0]) == 4:
        m = shape_dict['in_shape'][0][0] * shape_dict['in_shape'][0][1]
        k = shape_dict['in_shape'][0][2]
        n = shape_dict['out_shape'][0][2]
    elif len(shape_dict['in_shape'][0]) == 3:
        m = shape_dict['in_shape'][0][0]
        k = shape_dict['in_shape'][0][1]
        n = shape_dict['out_shape'][0][1]
    else:
        print(f"corner shape: {len(shape_dict['in_shape'][0])}")
    if n == 12:
        continue
    config[pytorch_name] = {'tesa_id': str(tesa_id), 'm': m, 'k': k, 'n': n}

pattern = "hubert_coarse_int8"

result = generate_code(config, pattern)

with open("kernel_dict.json", "w") as outfile:
    json.dump(result, outfile)