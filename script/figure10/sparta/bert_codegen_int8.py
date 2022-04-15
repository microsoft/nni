from tkinter import N
import torch
import json
from sparta.artifact_specialization import generate_code

tesaid_2_names_file = "artifact_bert_mixed_onnx_with_tesa/tesaid_2_names"
tesaid_2_names = torch.load(tesaid_2_names_file)
config = {}

id_shapes_name = "id_shapes"
f = open(id_shapes_name)
id_shapes = json.load(f)

for tesa_id, name_list in tesaid_2_names.items():
    pytorch_name, onnx_name = name_list[0], name_list[1]
    shape_dict = id_shapes[str(tesa_id)]
    if len(shape_dict['in_shape'][0]) == 3:
        m = shape_dict['in_shape'][0][0] * shape_dict['in_shape'][0][1]
        k = shape_dict['in_shape'][0][2]
        n = shape_dict['out_shape'][0][2]
    elif len(shape_dict['in_shape'][0]) == 2:
        m = shape_dict['in_shape'][0][0]
        k = shape_dict['in_shape'][0][1]
        n = shape_dict['out_shape'][0][1]
    config[pytorch_name] = {'tesa_id': str(tesa_id), 'm': m, 'k': k, 'n': n}

pattern = "bert_coarse_int8"

result = generate_code(config, pattern)

with open("kernel_dict.json", "w") as outfile:
    json.dump(result, outfile)