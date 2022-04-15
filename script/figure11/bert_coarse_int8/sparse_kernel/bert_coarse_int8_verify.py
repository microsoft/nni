from re import template
from tkinter import N
import torch
import json
from sparta.artifact_specialization import generate_code_verify
from numpy import array
from sparta.common.utils import convert_to_block_csr_int8, convert_to_block_csr_bin
from sparta.common.sparsity import TeSA
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple 

def kernel_execution(kernel: str) -> Tuple[str, float, bool]:
    # kernel execution process
    file_name_new = "kernel_generate_code.cu"
    with open(file_name_new, 'w') as f:
        f.write(kernel)
    avg_latency, success = run_gpu_kernel(file_name_new)
    # kernel correctness verification failure
    if success == False:
        avg_latency = 10000
    return kernel, avg_latency, success

def verify_successful(file_name):
    with open(file_name, 'r') as f:
        content = f.read()
    if content.find("Pass") == -1:
        return False
    return True

def get_kernel_run_time(file_name):
    lines = []
    kernel_name = "Time="
    with open(file_name, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.find(kernel_name) == -1:
            continue
        else:
            run_time = float(line.split()[-2])
            break
    return run_time

def run_gpu_kernel(file_name):
    compile_cmd = 'nvcc -gencode arch=compute_75,code=sm_75 \
    -L/usr/lib/x86_64-linux-gnu/ -lcudnn {} -o {} -O3 -std=c++11'.format(file_name, Path(file_name).stem)
    output_file_name = f"output_log.txt"
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=600)
    latencys = []
    for i in range(2):
        command = './{} > {}'.format(Path(file_name).stem, output_file_name)
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=600)

        if i == 0:
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_name)))
    success = verify_successful(output_file_name)
    avg_latency = sum(latencys) / len(latencys)
    print(f"avg_latency is {avg_latency}")
    print(f"success or not: {success}")
    return avg_latency, success

def write_array(data, file_path, dtype="i"):
    array_data = array(data, dtype)
    with open(file_path, 'wb') as f:
        array_data.tofile(f)

def bcsr_generate(weight_tesa: TeSA, bit: int):
    config = {}
    if weight_tesa.block_size is None:
        return config

    block_size_n, block_size_k = weight_tesa.block_size[0], weight_tesa.block_size[1]
    weight_rand = torch.rand(weight_tesa.tesa.shape)
    if bit == 32:
        rows, cols, vals = convert_to_block_csr_bin(weight_tesa.tesa.t(), weight_rand.t(), block_size_k, block_size_n)
    elif bit == 8:
        rows, cols, vals = convert_to_block_csr_int8(weight_tesa.tesa.t(), weight_rand.t(), block_size_k, block_size_n)

    # import ipdb; ipdb.set_trace()
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    
    write_array(rows, rows_path)
    write_array(cols, cols_path)
    write_array(vals, vals_path, "f")

    config['ROW_PATH_VALUE'] = '"' + rows_path + '"'
    config['COL_PATH_VALUE'] = '"' + cols_path + '"'
    config['VAL_PATH_VALUE'] = '"' + vals_path + '"'

    return config


def bert_coarse_int8_verify(config: dict):
    result = {}
    log_name = "../../Log/bert_coarse_int8_breakdown_add_sparse.json"
    template_name = "../../Template_test/block_quantize_template_bias.cu"
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(template_name)
    template_str = f_template.read()
    tesa_dict = torch.load("artifact_bert_coarse_no_propagation_onnx_with_tesa/tesa")
    for name, val_dict in config.items():
        tesa_id = val_dict['tesa_id']
        print(f"tesa_id: {tesa_id}")
        m, k, n = val_dict['m'], val_dict['k'], val_dict['n']
        print(f"m: {m}, k: {k}, n: {n}")
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
        print(f"block_size_m: {block_size_col}, block_size_n: {block_size_row}, block_size_k: {block_size_k}")

        kernel_code = template_str
        for key, value in template_config.items():
            kernel_code = kernel_code.replace(key, str(value))

        tesa_weight = tesa_dict[int(tesa_id)]['weight']
        tesa_weight = TeSA(tesa_weight)
        tesa_weight.set_transform_meta((block_size_row, block_size_k), 8)
        bcsr_config = bcsr_generate(tesa_weight, 8)
        for key, value in bcsr_config.items():
            kernel_code = kernel_code.replace(key, str(value))
        kernel, avg_latency, success = kernel_execution(kernel_code)
        if success == False:
            print(f"tesa id: {tesa_id}, module name: {name}, shape:{[m, k, n]}")
        #assert(int(block_size_n / thread_size_n) != 0 and int(block_size_m / thread_size_m) != 0)
        #assert(int(n / block_size_n) != 0 and int(m / block_size_m) != 0)

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

pattern = "bert_coarse_int8"

bert_coarse_int8_verify(config)