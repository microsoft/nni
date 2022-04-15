from encodings import search_function
from multiprocessing.sharedctypes import Value
from operator import mod
from sre_constants import SUCCESS
from tkinter import N

from click import launch
from numpy import array
from sparta.common.sparsity import TeSA
import torch
import json
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
import subprocess
from sparta.common.utils import convert_to_block_csr_int8, convert_to_block_csr_bin

current_path = Path(__file__).parent


support_patterns = ["mobilenet_coarse_int8", "bert_coarse_fp32", "bert_coarse_int8", "hubert_coarse_fp32", "hubert_coarse_int8"]

def write_array(data, file_path, dtype="i"):
    array_data = array(data, dtype)
    with open(file_path, 'wb') as f:
        array_data.tofile(f)

def convert_to_block_csr(m_tensor, v_tensor, block_h, block_w):
    assert len(m_tensor.size()) == 2
    size_h, size_w = m_tensor.size()
    if size_h % block_h != 0 or size_w % block_w != 0:
        return None, None, None

    rows = []
    cols = []
    values = []
    for _i in range(size_h//block_h):
        rows.append(len(cols))
        for _j in range(size_w//block_w):
            i_start = _i * block_h
            i_end = (_i+1) * block_h
            j_start = _j * block_w
            j_end = (_j+1) * block_w
            if torch.sum(m_tensor[i_start:i_end, j_start:j_end]) > 0:
                cols.append(_j)
                values.extend(v_tensor[i_start:i_end,j_start:j_end].flatten().tolist())
    rows.append(len(cols))
    return rows, cols, values

def bcsr_clean():
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    paths = [rows_path, cols_path, vals_path]
    for path in paths:
        os.remove(path)

def bcsr_generate(weight_tesa: TeSA, bit: int):
    config = {}
    if weight_tesa.block_size is None:
        return config

    block_size_n, block_size_k = weight_tesa.block_size[0], weight_tesa.block_size[1]
    if bit == 32:
        rows, cols, vals = convert_to_block_csr_bin(weight_tesa.tesa.t(), weight_tesa.tesa.t(), block_size_k, block_size_n)
    elif bit == 8:
        rows, cols, vals = convert_to_block_csr_int8(weight_tesa.tesa.t(), weight_tesa.tesa.t(), block_size_k, block_size_n)

    # import ipdb; ipdb.set_trace()
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    
    write_array(rows, rows_path)
    write_array(cols, cols_path)
    write_array(vals, vals_path, 'f')

    config['ROW_PATH_VALUE'] = '"' + rows_path + '"'
    config['COL_PATH_VALUE'] = '"' + cols_path + '"'
    config['VAL_PATH_VALUE'] = '"' + vals_path + '"'

    return config

def bcsr_clean():
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    paths = [rows_path, cols_path, vals_path]
    for path in paths:
        os.remove(path)

def generate_code_verify(config: dict, pattern: str) -> dict:
    assert(pattern in support_patterns, f"only support support_patterns: {support_patterns}")
    if pattern == "mobilenet_coarse_int8":
        mobilenet_coarse_int8_verify(config)
    elif pattern == "bert_coarse_fp32":
        bert_coarse_fp32_verify(config)
    elif pattern == "bert_coarse_int8":
        bert_coarse_int8_verify(config)
    elif pattern == "hubert_coarse_fp32":
        hubert_coarse_fp32_verify(config)
    elif pattern == "hubert_coarse_int8":
        hubert_coarse_int8_verify(config)

def generate_code(config: dict, pattern: str) -> dict:
    assert(pattern in support_patterns, f"only support support_patterns: {support_patterns}")
    if pattern == "mobilenet_coarse_int8":
        result = mobilenet_coarse_int8_codegen(config)
    elif pattern == "bert_coarse_fp32":
        result = bert_coarse_fp32_codegen(config)
    elif pattern == "bert_coarse_int8":
        result = bert_coarse_int8_codegen(config)
    elif pattern == "hubert_coarse_fp32":
        result = hubert_coarse_fp32_codegen(config)
    elif pattern == "hubert_coarse_int8":
        result = hubert_coarse_int8_codegen(config)
    return result

def bert_coarse_fp32_verify(config: dict):
    print(f"current_path: {current_path}")
    result = {}
    log_name = os.path.join(current_path, "Log/bert_coarse_fp32.json")
    template_name = os.path.join(current_path, "Template_test/block_sparse_template_bias_row.cu")
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(template_name)
    template_str = f_template.read()
    tesa_dict = torch.load("/Data/nizhen/SparTA/script/checkpoints/bert/artifact_bert_coarse_onnx_with_tesa/tesa")
    for name, val_dict in config.items():
        tesa_id = val_dict['tesa_id']
        print(f"tesa_id: {tesa_id}")
        m, k, n = val_dict['m'], val_dict['k'], val_dict['n']
        if tesa_id in log_dict:
            template_config = log_dict[tesa_id]
        else:
            template_config = log_dict["other"]
        template_config['M_VALUE'] = m
        template_config['K_VALUE'] = k
        template_config['N_VALUE'] = n
        if n <= template_config['BLOCK_SIZE_N_VALUE']:
            template_config['BLOCK_SIZE_N_VALUE'] = n
        if m <= template_config['BLOCK_SIZE_M_VALUE']:
            template_config['BLOCK_SIZE_M_VALUE'] = m
        template_config['BLOCK_SIZE_N_VALUE'] = 64
        kernel_code = template_str
        for key, value in template_config.items():
            kernel_code = kernel_code.replace(key, str(value))

        tesa_weight = tesa_dict[int(tesa_id)]['weight']
        tesa_weight = TeSA(tesa_weight)
        tesa_weight.set_transform_meta((template_config['BLOCK_SIZE_N_VALUE'], template_config['BLOCK_SIZE_K_VALUE']), 32)
        bcsr_config = bcsr_generate(tesa_weight, 32)
        for key, value in bcsr_config.items():
            kernel_code = kernel_code.replace(key, str(value))
        kernel, avg_latency, success = kernel_execution(kernel_code)
        if success == False:
            print(f"tesa id: {tesa_id}, module name: {name}, shape:{[m, k, n]}")
        #assert(int(block_size_n / thread_size_n) != 0 and int(block_size_m / thread_size_m) != 0)
        #assert(int(n / block_size_n) != 0 and int(m / block_size_m) != 0)

def mobilenet_coarse_int8_verify(config: dict):
    ...

def bert_coarse_int8_verify(config: dict):
    print(f"current_path: {current_path}")
    result = {}
    log_name = os.path.join(current_path, "Log/bert_coarse_int8.json")
    template_name = os.path.join(current_path, "Template_test/block_quantize_template_bias.cu")
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(template_name)
    template_str = f_template.read()
    tesa_dict = torch.load("/Data/nizhen/SparTA/script/checkpoints/bert/artifact_bert_coarse_onnx_with_tesa/tesa")
    for name, val_dict in config.items():
        tesa_id = val_dict['tesa_id']
        print(f"tesa_id: {tesa_id}")
        if int(tesa_id) != 73:
            continue
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

def hubert_coarse_fp32_verify(config: dict):
    ...

def hubert_coarse_int8_verify(config: dict):
    ...

def mobilenet_coarse_int8_codegen(config: dict) -> dict:
    print(f"current_path: {current_path}")
    result = {}
    log_name = os.path.join(current_path, "Log/mobilenet_coarse_int8.json")
    conv_one_template_name = os.path.join(current_path, "Template/quantize_dot_template_bias.cu")
    f = open(log_name)
    log_dict = json.load(f)
    f_template = open(conv_one_template_name)
    conv_one_template_str = f_template.read()
    depth_one_template_name = os.path.join(current_path, "Template/depth_quantize_temlate_bias.cu")
    f_template = open(depth_one_template_name)
    depth_conv_template_str = f_template.read()

    for name, val_dict in config.items():
        op_type = config[name]['op_type']
        if op_type == "conv1x1":
            m, k, n = val_dict['m'], val_dict['k'], val_dict['n']
            log_key = f"{m}_{k}_{n}"
            if log_key in log_dict:
                print(f"{log_key} find in log")
                template_config = log_dict[log_key]
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
            block_num = int((m * n) / (block_size_col * block_size_row))
            thread_num = block_row_warps * block_col_warps * 32
            kernel_code = conv_one_template_str
            for k, v in template_config.items():
                kernel_code = kernel_code.replace(k, str(v))
            launch_config = {}
            launch_config['dimBlock'] = [thread_num, 1]
            launch_config['dimGrid'] = [block_num,1]
            result[name] = {'code': kernel_code, 'launch_config': launch_config}

        elif op_type == "depth_conv":
            template_config = {}
            template_config['CHANNEL_VALUE'] = val_dict['channel']
            template_config['IN_HEIGHT_VALUE'] = val_dict['in_height']
            template_config['IN_WIDTH_VALUE'] = val_dict['in_width']
            template_config['OUT_HEIGHT_VALUE'] = val_dict['out_height']
            template_config['OUT_WIDTH_VALUE'] = val_dict['out_width']
            template_config['BATCHSIZE_VALUE'] = val_dict['batch_size']
            template_config['KERNEL_H_VALUE'] = val_dict['kernel_h']
            template_config['KERNEL_W_VALUE'] = val_dict['kernel_w']
            template_config['STRIDE_H_VALUE'] = val_dict['stride_h']
            template_config['STRIDE_W_VALUE'] = val_dict['stride_w']
            template_config['PAD_H_VALUE'] = val_dict['pad_h']
            template_config['PAD_W_VALUE'] = val_dict['pad_w']
            kernel_code = depth_conv_template_str
            for k, v in template_config.items():
                kernel_code = kernel_code.replace(k, str(v))

            launch_config = {}
            out_count = int(template_config['BATCHSIZE_VALUE'] * template_config['CHANNEL_VALUE']\
                 * template_config['OUT_HEIGHT_VALUE'] * template_config['OUT_WIDTH_VALUE'])
            thread_num = 512
            block_num = int(out_count / thread_num)
            launch_config['dimBlock'] = [thread_num, 1]
            launch_config['dimGrid'] = [block_num, 1]

            result[name] = {'code': kernel_code, 'launch_config': launch_config}
    return result

def bert_coarse_fp32_codegen(config: dict) -> dict:
    print(f"current_path: {current_path}")
    result = {}
    log_name = os.path.join(current_path, "Log/bert_coarse_fp32.json")
    template_name = os.path.join(current_path, "Template/block_sparse_template_bias_row.cu")
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
            template_config['BLOCK_SIZE_N_VALUE'] = n
        if m <= template_config['BLOCK_SIZE_M_VALUE']:
            template_config['BLOCK_SIZE_M_VALUE'] = m
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

def bert_coarse_int8_codegen(config: dict) -> dict:
    result = {}
    log_name = os.path.join(current_path, "Log/bert_coarse_int8.json")
    template_name = os.path.join(current_path, "Template/block_quantize_template_bias.cu")
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

def hubert_coarse_fp32_codegen(config: dict) -> dict:
    ...

def hubert_coarse_int8_codegen(config: dict) -> dict:
    result = {}
    log_name = os.path.join(current_path, "Log/hubert_coarse_int8.json")
    template_name = os.path.join(current_path, "Template/quantize_dot_template_bias.cu")
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

def verify_successful(file_name):
    with open(file_name, 'r') as f:
        content = f.read()
    if content.find("Pass") == -1:
        return False
    return True