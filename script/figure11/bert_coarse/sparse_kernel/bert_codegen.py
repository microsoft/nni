from tkinter import N
import torch
import json
import os
from sparta.artifact_specialization import generate_code

tesaid_2_names_file = "artifact_bert_coarse_no_propagation_onnx_with_tesa/tesaid_2_names"
tesaid_2_names = torch.load(tesaid_2_names_file)
config = {}

id_shapes_name = "artifact_bert_coarse_no_propagation_onnx_with_tesa/shape.json"
f = open(id_shapes_name)
id_shapes = json.load(f)

def bert_coarse_fp32_codegen(config: dict) -> dict:
    result = {}
    log_name = "../../Log/bert_coarse_fp32_breakdown_add_sparse.json"
    template_name = "../../Template/block_sparse_template_bias_row.cu"
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(template_name)
    template_str = f_template.read()
    for name, val_dict in config.items():
        tesa_id = val_dict['tesa_id']
        m, k, n = val_dict['m'], val_dict['k'], val_dict['n']
        if tesa_id in log_dict:
            template_config = log_dict[tesa_id].copy()
        else:
            template_config = log_dict["other"].copy()
        template_config['M_VALUE'] = m
        template_config['K_VALUE'] = k
        template_config['N_VALUE'] = n
        if n <= template_config['BLOCK_SIZE_N_VALUE']:
            template_config['BLOCK_SIZE_N_VALUE'] = n-1
        if m <= template_config['BLOCK_SIZE_M_VALUE']:
            template_config['BLOCK_SIZE_M_VALUE'] = m-1
        block_size_m = template_config['BLOCK_SIZE_M_VALUE']
        block_size_k = template_config['BLOCK_SIZE_K_VALUE']
        block_size_n = template_config['BLOCK_SIZE_N_VALUE']
        thread_size_m = template_config['THREAD_SIZE_M_VALUE']
        thread_size_k = template_config['THREAD_SIZE_K_VALUE']
        thread_size_n = template_config['THREAD_SIZE_N_VALUE']
        kernel_code = template_str
        for key, value in template_config.items():
            kernel_code = kernel_code.replace(key, str(value))
        launch_config = {}
        launch_config['dimBlock'] = [int(block_size_n / thread_size_n), int(block_size_m / thread_size_m)]
        launch_config['dimGrid'] = [int(n / block_size_n), int(m / block_size_m)]

        print(f"tesa id: {tesa_id}, module name: {name}, launch_config: {launch_config}, shape:{[m, k, n]}")

        #assert(int(block_size_n / thread_size_n) != 0 and int(block_size_m / thread_size_m) != 0)
        #assert(int(n / block_size_n) != 0 and int(m / block_size_m) != 0)

        result[name] = {'code': kernel_code, 'launch_config': launch_config, 'block_size_k': block_size_k, 'block_size_n': block_size_n}
    return result

tesa_pytorch_name_list = []
for tesa_id, name_list in tesaid_2_names.items():
    pytorch_name = name_list[0]
    tesa_pytorch_name_list.append(pytorch_name)

for tesa_id, name_list in tesaid_2_names.items():
    pytorch_name, onnx_name = name_list[0], name_list[1]
    shape_dict = id_shapes[str(pytorch_name)]
    if pytorch_name not in tesa_pytorch_name_list:
        continue
    #import ipdb; ipdb.set_trace()
    if len(shape_dict['in_shape'][0]) == 4:
        m = shape_dict['in_shape'][0][0] * shape_dict['in_shape'][0][1]
        k = shape_dict['in_shape'][0][2]
        n = shape_dict['out_shape'][0][2]
    elif len(shape_dict['in_shape'][0]) == 3:
        m = shape_dict['in_shape'][0][0]
        k = shape_dict['in_shape'][0][1]
        n = shape_dict['out_shape'][0][1]
    else:
        NotImplementedError("not support shape")
    config[pytorch_name] = {'tesa_id': str(tesa_id), 'm': m, 'k': k, 'n': n}

result = bert_coarse_fp32_codegen(config)

with open("kernel_dict.json", "w") as outfile:
    json.dump(result, outfile)