from tkinter import N
import torch
import json
from sparta.artifact_specialization import generate_code
import os

def bert_coarse_int8_codegen(config: dict) -> dict:
    result = {}
    log_name = "../../Log/bert_coarse_int8_breakdown_add_sparse.json"
    template_name = "../../Template/block_quantize_template_bias.cu"
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(template_name)
    template_str = f_template.read()
    for name, val_dict in config.items():
        tesa_id = val_dict['tesa_id']
        m, k, n = val_dict['m'], val_dict['k'], val_dict['n']
        if tesa_id in log_dict:
            template_config = log_dict[tesa_id]
        else:
            template_config = log_dict["other"]
        template_config['M_VALUE'] = m
        template_config['K_VALUE'] = k
        template_config['N_VALUE'] = n
        block_col_warps = template_config['BLOCK_COL_WARPS_VALUE']
        block_row_warps = template_config['BLOCK_ROW_WARPS_VALUE']
        warp_col_tiles = template_config['WARP_COL_TILES_VALUE']
        warp_row_tiles = template_config['WARP_ROW_TILES_VALUE']
        block_size_col = block_col_warps * warp_col_tiles * 16
        block_size_row = block_row_warps * warp_row_tiles * 16
        block_size_k = template_config['CHUNK_K_VALUE'] * 16
        if block_size_k > k:
            template_config['CHUNK_K_VALUE'] = int(template_config['CHUNK_K_VALUE'] / 2)
            block_size_k = template_config['CHUNK_K_VALUE'] * 16
        block_num = int((m * n) / (block_size_col * block_size_row))
        thread_num = block_row_warps * block_col_warps * 32
        kernel_code = template_str
        for k, v in template_config.items():
            kernel_code = kernel_code.replace(k, str(v))
        launch_config = {}
        launch_config['dimBlock'] = [thread_num, 1]
        launch_config['dimGrid'] = [block_num,1]
        result[name] = {'code': kernel_code, 'launch_config': launch_config, 'block_size_n': block_size_row, 'block_size_k': block_size_k}
    return result

tesaid_2_names_file = "artifact_bert_coarse_no_propagation_onnx_with_tesa/tesaid_2_names"
tesaid_2_names = torch.load(tesaid_2_names_file)
config = {}

id_shapes_name = "artifact_bert_coarse_no_propagation_onnx_with_tesa/shape.json"
f = open(id_shapes_name)
id_shapes = json.load(f)

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

result = bert_coarse_int8_codegen(config)

with open("kernel_dict.json", "w") as outfile:
    json.dump(result, outfile)