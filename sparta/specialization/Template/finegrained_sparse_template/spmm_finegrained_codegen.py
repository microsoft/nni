'''
dense(C) = sparse(A) x dense(B)
'''

import copy
from dataclasses import dataclass
import os
from pathlib import Path
import random
import subprocess
from torch import device

from code_template_short_gpu_new_fast import _GpuCodeTemplate, _GpuBodyCodeTemplate, _GpuBodyCodePruneWeightTemplate
# from code_template_short_gpu_new import _GpuCodeTemplate, _GpuBodyCodeTemplate, _GpuBodyCodePruneWeightTemplate
from generate_matrix import *

import nni


M = 1024
K = 1024
N = 1024
TESAID = 0
sp_matrix = None
SPARSITY = 99
DATA_FOLDER = 'data'
CODE_FOLDER = 'codehub'
FILENAME = 'sparse_matrix.txt'
tiling_str = None
matrix_str = None
DEVICE_FUNC_PREFIX = None
device_funcs = []

BLOCK_SIZE_M = 0
BLOCK_SIZE_K = 0
BLOCK_SIZE_N = 0

def emit_block_device_func(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, blockIdx_x, DEVICE_FUNC_PREFIX):
    device_func = ""

    device_func_sig = f'__forceinline__ __device__ void {DEVICE_FUNC_PREFIX}_device_func_blockIdx_x_{blockIdx_x}(float* input0, float *rC) {{\n'
    device_func += device_func_sig

    # device_func += "float* input0_tile = input0;\n"
    num_K_tiles = int(K1 / K2)
    for k_tile_idx in range(num_K_tiles):
        # device_func += f"input0_tile = input0 + blockIdx.x * {N2} + {k_tile_idx * K2 * N1};\n"
        for m_step in range(M2):
            for k_step in range(K2):
                tesa_value = tesa_matrix[(blockIdx_x * M2 + m_step) * K1 + k_tile_idx * K2 + k_step]
                sp_matrix_value = sp_matrix[(blockIdx_x * M2 + m_step) * K1 + k_tile_idx * K2 + k_step]
                if tesa_value != 0:
                    device_func += f"rC[{m_step}] += {sp_matrix_value}f * input0[blockIdx.y * {N2} + {(k_tile_idx * K2 + k_step) * N1} + threadIdx.x];\n"

    device_func += f"}}\n"

    return device_func




def emit_finegrained_sparse_kernel_body(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, DEVICE_FUNC_PREFIX, TRANSPOSE_OUTPUT=False, FUSE_ADD=False):
    device_funcs = []
    func_body = ""

    # emit device_funcs
    blockDim_x = int(M1 / M2)
    for blockIdx_x in range(blockDim_x):
        device_func = emit_block_device_func(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, blockIdx_x, DEVICE_FUNC_PREFIX)
        # print(device_func)
        device_funcs.append(device_func)

    emit_local_alloc = f"float output0_local[{M2}] = {{0}};\n"
    func_body += emit_local_alloc

    for blockIdx_x in range(blockDim_x):
        func_body += f"if (blockIdx.x == {blockIdx_x}) {{\n"
        func_body += f"{DEVICE_FUNC_PREFIX}_device_func_blockIdx_x_{blockIdx_x}(input0, output0_local);\n"
        func_body += f"}}\n"

    if not TRANSPOSE_OUTPUT:
        if FUSE_ADD:
            func_body += f"float *output0_tile = output0 + (blockIdx.x * {M2}) * {N1} + blockIdx.y * {N2};\n"
            for m_step in range(M2):
                func_body += f"output0_tile[{m_step * N1} + threadIdx.x] = output0_local[{m_step}] + input1[blockIdx.x * {N2} + threadIdx.x];\n"
        else:
            func_body += f"float *output0_tile = output0 + (blockIdx.x * {M2}) * {N1} + blockIdx.y * {N2};\n"
            for m_step in range(M2):
                func_body += f"output0_tile[{m_step * N1} + threadIdx.x] = output0_local[{m_step}];\n"
    else:
        pass
    

    # # write back output0_local
    # func_body += f"float *output0_tile = output0 + (blockIdx.x * {M2} + threadIdx.x) * {N1} + blockIdx.y * {N2};\n"
    # func_body += f"float4 *output0_tile_f4 = reinterpret_cast<float4*>(output0_tile);\n"
    # func_body += f"float4 *output0_local_f4 = reinterpret_cast<float4*>(output0_local);\n"
    # if FUSE_ADD:
    #     func_body += f"float4 *bias_f4 = reinterpret_cast<float4*>(input2+blockIdx.y*{N2});\n"
    #     func_body += f"float4 bias_f4_local;\n"
    #     for item in range(int(N2/4)):
    #         func_body += f"bias_f4_local = bias_f4[{item}];\n"
    #         func_body += f"output0_local_f4[{item}].x += bias_f4_local.x;\n"
    #         func_body += f"output0_local_f4[{item}].y += bias_f4_local.y;\n"
    #         func_body += f"output0_local_f4[{item}].z += bias_f4_local.z;\n"
    #         func_body += f"output0_local_f4[{item}].w += bias_f4_local.w;\n"
    
    # for item in range(int(N2/4)):
    #     func_body += f"output0_tile_f4[{item}] = output0_local_f4[{item}];\n"

    grid_dim = (int(M1/M2), int(N1/N2), 1)
    block_dim = (N2, 1, 1)

    return func_body, device_funcs, grid_dim, block_dim

def emit_finegrained_sparse_kernel_entry(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, TESAID, TRANSPOSE_OUTPUT=False, FUSE_ADD=False):
    DEVICE_FUNC_PREFIX = ""
    if FUSE_ADD:
        DEVICE_FUNC_PREFIX = f"Fused_Dot_Add_sparta_finegrained_TESAID_{TESAID}_{M1}_{K1}_{N1}_{M2}_{K2}_{N2}"
    else:
        DEVICE_FUNC_PREFIX = f"Dot_sparta_finegrained_TESAID_{TESAID}_{M1}_{K1}_{N1}_{M2}_{K2}_{N2}"

    func_body, device_funcs, grid_dim, block_dim = emit_finegrained_sparse_kernel_body(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, DEVICE_FUNC_PREFIX, TRANSPOSE_OUTPUT, FUSE_ADD)

    _FunctionBodyCodeTemplate = '''
__global__ void sparta_finegrained_dot_kernel0(float *input0, float *input1, float *output0)
{{
    {}
}}
'''

    _FusedFunctionBodyCodeTemplate = '''
__global__ void sparta_fused_finegrained_dot_add_kernel0(float *input0, float *input1, float *input2, float *output0)
{{
    {}
}}
'''

    kernel_entry = {}

    dev_func = ""
    for item in device_funcs:
        dev_func += item

    kernel_entry["function_comment"] = dev_func

    if FUSE_ADD:
        kernel_entry["code"] = _FusedFunctionBodyCodeTemplate.format(func_body)
        kernel_entry["op_type"] = "Fused_Dot_Add"
    else:
        kernel_entry["code"] = _FunctionBodyCodeTemplate.format(func_body)
        kernel_entry["op_type"] = "Dot"

    kernel_entry["miscs"] = {"TESAID": TESAID, "SparTA": {"Weight_Dismantled": True}}
    kernel_entry["tvm_func_name"] = DEVICE_FUNC_PREFIX + "_kernel0"

    kernel_entry["gridDim"] = grid_dim
    kernel_entry["blockDim"] = block_dim

    kernel_entry["parameters"] = {"arg0_shape": [M, K], "arg1_shape": [K, N], "out_shape": [M, N]}

    return kernel_entry



def generate_dense_schema(M1, N1, K1, M2, N2, K2):
    global BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N
    BLOCK_SIZE_M = M2
    BLOCK_SIZE_K = K2
    BLOCK_SIZE_N = N2


def code_gen(M1, N1, K1, M2, N2, K2, write_file=False):
    global sp_matrix, device_funcs, DEVICE_FUNC_PREFIX
    sp_matrix = load_tesa_matrix_float(FILENAME)
    generate_dense_schema(M1, N1, K1, M2, N2, K2)
    tesa_matrix = sp_matrix
    func_body, device_funcs, grid_dim, block_dim = emit_finegrained_sparse_kernel_body(tesa_matrix, sp_matrix, M1, N1, K1, M2, N2, K2, DEVICE_FUNC_PREFIX, False, False)

    bPerGrid = f"{grid_dim[0]}, {grid_dim[1]}, {grid_dim[2]}"
    tPerBlock = f"{block_dim[0]}, {block_dim[1]}, {block_dim[2]}"
    device_func_code = ""
    for device_func in device_funcs:
        device_func_code += device_func
    full_code = _GpuCodeTemplate.format(M, N, K, matrix_str, bPerGrid, tPerBlock, device_func_code, func_body)

    code_file_name = None
    if write_file:
        code_file_name = f'{CODE_FOLDER}/_generated_gpu_embed_{DEVICE_FUNC_PREFIX}_{tiling_str}.cu'
        with open(code_file_name, 'w') as fp:
            fp.write(full_code)

    return code_file_name


def _get_latency_num(file_name):
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            if 'matrixMultiplication' in line:
                segs = line.split()
                if segs[3][-2:] == 'us':
                    return float(segs[3][:-2])
                elif segs[3][-2:] == 'ms':
                    return float(segs[3][:-2]) * 1000
                else:
                    return 1000000000


def profile_gpu_code(code_file, tiling_args='manual'):
    compile_cmd = f'nvcc -gencode arch=compute_75,code=sm_75 -L/usr/lib/x86_64-linux-gnu/ -lcudnn {code_file} -o {CODE_FOLDER}/{Path(code_file).stem} -O3 -std=c++11'
    print(compile_cmd)
    #subprocess.check_output(compile_cmd, shell = True, text = True)
    os.system(compile_cmd)
    # exit(0)
    #result = subprocess.check_output(f'./{Path(code_file).stem}')
    # print(result)
    latencys = []
    for i in range(4):
        os.system(
            f'nvprof --unified-memory-profiling off ./{CODE_FOLDER}/{Path(code_file).stem} 2> {CODE_FOLDER}/a_{tiling_args}.txt')
        if i == 0:
            continue
        latencys.append(_get_latency_num(f'{CODE_FOLDER}/a_{tiling_args}.txt'))
    avg_latency = sum(latencys) / len(latencys)
    print(latencys)
    print(avg_latency)
    return avg_latency, latencys


if __name__ == '__main__':
    # ta = nni.get_next_parameter()
    ta = {'M2': 32, 'N2': 128, 'K2': 2}
    tiling_str = f'{ta["M2"]}_{ta["N2"]}_{ta["K2"]}'
    matrix_str = f'{M}_{K}_{N}'
    os.system(f'mkdir {CODE_FOLDER}')
    os.system(f'cp dev_array.h {CODE_FOLDER}/dev_array.h')
    os.system(f'mkdir {DATA_FOLDER}')
    FILENAME = f'{DATA_FOLDER}/sparse_matrix_{matrix_str}.txt'
    # FILENAME = f'{DATA_FOLDER}/sparse_matrix_{tiling_str}.txt'
    # FILENAME = f'{DATA_FOLDER}/mlp_mnist/sparse_matrix_{matrix_str}_fc4.txt'
    # FILENAME = f'{DATA_FOLDER}/mlp_mnist/sparse_matrix_{matrix_str}_{SPARSITY}_mix.txt'
    # FILENAME = f'../models/bert_finegrained_onnx_with_tesa/Constants/Dot_4096_768_768_TESAID_10.csv'
    generate_sparse_matrix_float(M, K, SPARSITY, FILENAME)
    DEVICE_FUNC_PREFIX = f'sparse_matrix_{matrix_str}_{SPARSITY}_TESAID_{TESAID}_bert_osdi_final'
    code_file = code_gen(M, N, K, ta["M2"], ta["N2"], ta["K2"], write_file=True)
    avg_latency, latencys = profile_gpu_code(code_file, tiling_str)
    # nni.report_final_result({'default': avg_latency, 'all_latency': latencys})
    # nni.report_final_result(avg_latency)



# bert_osdi 3072x768x4096_95 641us
# K2: 16
# M2: 64
# N2: 128

# bert_osdi 768x768x4096_95 182us
# K2: 1
# M2: 64
# N2: 128

# bert_osdi 768x3072x4096_95 758us
# K2: 1
# M2: 32
# N2: 256