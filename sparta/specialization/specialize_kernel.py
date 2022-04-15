import itertools
import math
import torch
import subprocess
import os
from numpy import array
from sqlite3 import paramstyle
from pytest import param
from sparta.common.sparsity import TeSA, TesaAttr
# from ..common.sparsity import TeSA, TesaAttr
from typing import Dict, List, Optional, Tuple
from pathlib import Path

current_path = Path(__file__).parent

__all__ = ['specialize_matmul']

def specialize_matmul(in_tesa: tuple, weight_tesa: tuple, out_t_tesa: tuple):
    """
    Generate the kernels and profile the combined latency
    """
    # mocked for now
    assert(len(in_tesa) == 1 and len(out_t_tesa) == 1 and len(weight_tesa) >= 1)

    i_tesa, o_tesa = in_tesa[0], out_t_tesa[0]
    
    latency = 0
    kernels = []
    # for matmul, aggr_type should be add
    aggr_type = 'add'
    for idx, w_tesa in enumerate(weight_tesa):
        if idx == 0:
            kernel, kernel_test, params = matmul_kernel_init(i_tesa, w_tesa, o_tesa)
        else:
            kernel, kernel_test, params = matmul_kernel_init(i_tesa, w_tesa, o_tesa, bias=True)
        _, exec_time, best_param_dict = matmul_kernel_tune(kernel_test, params, w_tesa)
        for key, val in best_param_dict.items():
            kernel = kernel.replace(key, str(int(val)))
        kernels.append(kernel)
        latency += exec_time

    return latency, kernels, aggr_type

def matmul_kernel_codegen(w_tesa, shape: List[int], dismantle: int, block_size: tuple, n_bits: int, bias: bool) -> Tuple[str, str, Dict[str, List[int]]]:
    """
    Choose kernel template according to dismantle primitive
    """
    if dismantle == -1:
        kernel, kernel_test, params = dense_matmul_template(shape, n_bits, bias)
    elif dismantle == 8:
        kernel, kernel_test, params = finegrained_matmul_template(w_tesa, shape, n_bits, bias)
    elif dismantle == 2:
        kernel, kernel_test, params = blocksparse_matmul_template(shape, n_bits, block_size, bias)
    return kernel, kernel_test, params

def matmul_kernel_init(i_tesa: TeSA, w_tesa: TeSA, o_tesa: TeSA, bias=False) -> Tuple[str, str, Dict[str, List[int]]]:
    """
    Generate basic kernel code based on matmul shape, dismantle primitive, block size and bits.
    i, j, k -> i0, j0, k0, k1, i1, j1, k2, i2, j2
    dense: dismantle = -1, it will not be applied to any loop.
    finegrained sparse: dismantle = 8, it will be applied to the inner most loop.
    block sparse: dismantle = 2, it will be applied to k0.

    """
    matmul_shape = get_matmul_shape(i_tesa.tesa, w_tesa.tesa, o_tesa.tesa)
    block_size = w_tesa.block_size
    n_bits = w_tesa.n_bits
    if block_size == None:
        dismantle = -1
    elif block_size == 1:
        # not apply dismantle, kernel execute in dense way.
        dismantle = 8
    else:
        dismantle = 2

    kernel, kernel_test, params = matmul_kernel_codegen(w_tesa, matmul_shape, dismantle, block_size, n_bits, bias)
    return kernel, kernel_test, params

def write_array(data, file_path, dtype="i"):
    array_data = array(data, dtype)
    with open(file_path, 'wb') as f:
        array_data.tofile(f)

def bcsr_clean():
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    paths = [rows_path, cols_path, vals_path]
    for path in paths:
        os.remove(path)

def bcsr_generate(weight_tesa: TeSA):
    config = {}
    if weight_tesa.block_size is None:
        return config

    block_size_n, block_size_k = weight_tesa.block_size[0], weight_tesa.block_size[1]
    rows, cols, vals = convert_to_block_csr(weight_tesa.tesa, weight_tesa.tesa, block_size_n, block_size_k)
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    
    write_array(rows, rows_path)
    write_array(cols, cols_path)
    write_array(vals, vals_path)

    config['ROW_PATH_VALUE'] = '"' + rows_path + '"'
    config['COL_PATH_VALUE'] = '"' + cols_path + '"'
    config['VAL_PATH_VALUE'] = '"' + vals_path + '"'

    return config

def matmul_kernel_tune(kernel, params, weight_tesa):
    """
    Kernel tuning process
    """
    iters_list, dict_keys = generate_grid_search_space(params)
    # search_space = generate_grid_search_space(params)
    least_exec_time = math.inf
    best_kernel = None
    best_iters = None
    bcsr_config = bcsr_generate(weight_tesa)
    # embed bcsr path into code
    for key, val in bcsr_config.items():
        kernel = kernel.replace(key, str(val))

    original_kernel = kernel
    for iters in iters_list:
        kernel, exec_time = kernel_execution(original_kernel, iters, dict_keys)
        if exec_time < least_exec_time:
            best_kernel = kernel
            least_exec_time = exec_time
            best_iters = iters
    best_param_dict = {}
    assert(len(iters) == len(dict_keys))
    for i, val in enumerate(best_iters):
        best_param_dict[dict_keys[i]] = val
    
    bcsr_clean()

    return best_kernel, least_exec_time, best_param_dict

def dense_matmul_template(shape, n_bits, bias):
    assert(n_bits == 8 or n_bits == 32, "only support two bit types currently")
    assert(len(shape) == 3, "shape should contain m, k, n")
    m, k, n = shape[0], shape[1], shape[2]
    if n_bits == 8:
        template_name = os.path.join(current_path, "Template/quantize_dot_template_bias.cu")
        f = open(template_name)
        kernel = f.read()
        sub_param = {"M_GLOBAL_VALUE": m, "K_GLOBAL_VALUE": k, "N_GLOBAL_VALUE": n}
        tuning_param = {"CHUNK_K_VALUE": [1, 2, 4, 6, 8], "BLOCK_ROW_WARPS_VALUE": [1, 2, 3, 4], "BLOCK_COL_WARPS_VALUE": [1, 2, 3, 4], "WARP_ROW_TILES_VALUE": [1, 2, 3, 4], "WARP_COL_TILES_VALUE": [1, 2, 3, 4]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(int(value)))
        
        ## add kernel for testing
        test_template_name = os.path.join(current_path, "Template_test/quantize_dot_template_bias.cu")
        f = open(test_template_name)
        kernel_test = f.read()
        for key, value in sub_param.items():
            kernel_test = kernel_test.replace(key, str(int(value)))
    else:
        # n_bits == 32, should use cublas directly
        NotImplementedError("Shouldn't get through the fp32 dense specialization pass.")
        pass
    return kernel, kernel_test, tuning_param

def finegrained_matmul_template(w_tesa, shape, n_bits, bias) -> Tuple[str, Dict[str, List[int]]]:
    """
    Parameters
    ----------
    w_tesa
        TeSA of weight
    shape
        List, contains [m, k, n]
    n_bits
        bit width of weight, Choice[8, 32]
    bias
        bool, true represent using add fusion, false represent using non-fusion
    
    Returns
    -------
    kernel: str
        kernel string, should have embedded m, k, n into itself
    tuning_param: dict
        dict of tuning parameters

    Return Parameters Usage Example
    -------
        kernel_tmp = kernel
        tuning_param = {"BLOCK_SIZE_M": [32, 64, 128], "BLOCK_SIZE_N": [32, 64, 128]}
        search_space = grid_search_space_gen(tuning_param)

        best_kernel = ""
        least_latency = math.inf

        for case in search_space:
            for key, val in case.items():
                kernel_tmp = kernel.replace(key, val)
            latency = exec_kernel(kernel_tmp)
            if latency < least_latency:
                least_latency = latency
                best_kernel = kernel_tmp
    """
    kernel = ""
    kernel_test = ""
    tuning_param = {}
    return kernel, kernel_test, tuning_param

def blocksparse_matmul_template(shape, n_bits, block_size, bias):
    assert(n_bits == 8 or n_bits == 32, "only support two bit types currently")
    assert(len(shape) == 3, "shape should contain m, k, n")

    m, k, n = shape[0], shape[1], shape[2]
    # block_size should in formation [block_size_k, block_size_n]
    block_size_k = block_size[0]
    block_size_n = block_size[1]

    if n_bits == 32:
        template_name = os.path.join(current_path, "Template/block_sparse_template_bias_row.cu")
        f = open(template_name)
        kernel = f.read()
        sub_param = {"M_GLOBAL_VALUE": m , "K_GLOBAL_VALUE": k, "N_GLOBAL_VALUE": n, \
            "BLOCK_SIZE_K_VALUE": block_size_k, "BLOCK_SIZE_N_VALUE": block_size_n}
        tuning_param = {"BLOCK_SIZE_M_VALUE": [32, 64, 128], "THREAD_SIZE_M_VALUE": [2, 4, 8, 16],\
            "THREAD_SIZE_K_VALUE": [1, 2, 4, 8], "THREAD_SIZE_N_VALUE": [2, 4, 8, 16]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(int(value)))
        
        # add test template
        test_template_name = os.path.join(current_path, "Template_test/block_sparse_template_bias_row.cu")
        f = open(test_template_name)
        kernel_test = f.read()
        for key, value in sub_param.items():
            kernel_test = kernel_test.replace(key, str(int(value)))
    else:
        template_name = os.path.join(current_path, "Template/block_quantize_template_bias.cu")
        f = open(template_name)
        kernel = f.read()
        chunk_k = block_size_k / 16
        warp_row_tiles = 1 if block_size_n <= 16 else 2
        block_row_warps = (block_size_n / 16) / warp_row_tiles
        
        sub_param = sub_param = {"M_GLOBAL_VALUE": m, "K_GLOBAL_VALUE": k, "N_GLOBAL_VALUE": n, \
            "CHUNK_K_VALUE": chunk_k, "BLOCK_ROW_WARPS_VALUE": block_row_warps, \
                "WARP_ROW_TILES_VALUE": warp_row_tiles}
        tuning_param = {"BLOCK_COL_WARPS_VALUE": [1, 2, 3, 4], "WARP_COL_TILES_VALUE": [1, 2, 3, 4]}
        for key, value in sub_param.items():
            kernel = kernel.replace(key, str(int(value)))
        
        # add test template
        test_template_name = os.path.join(current_path, "Template_test/block_quantize_template_bias.cu")
        f = open(test_template_name)
        kernel_test = f.read()
        for key, value in sub_param.items():
            kernel_test = kernel_test.replace(key, str(int(value)))
    
    return kernel, kernel_test, tuning_param

def get_matmul_shape(i_tesa, w_tesa, o_tesa):
    assert(len(i_tesa.shape) == 2 and len(w_tesa.shape) == 2 and len(o_tesa.shape) == 2)
    m = i_tesa.shape[0]
    k = i_tesa.shape[1]
    n = w_tesa.shape[1]

    shape = [m, k, n]
    return shape

def generate_grid_search_space(params: dict):
    val_list = [val for _, val in params.items()]
    iters = itertools.product(*val_list)
    iters_list = list(iters)
    return iters_list, list(params.keys())

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

def kernel_execution(kernel: str, iters: list, dict_keys: list) -> Tuple[str, float]:
    # kernel execution process
    file_name_new = "kernel_generate_code.cu"
    for i, val in enumerate(iters):
        key = dict_keys[i]
        kernel = kernel.replace(key, str((val)))
    with open(file_name_new, 'w') as f:
        f.write(kernel)
    avg_latency, success = run_gpu_kernel(file_name_new)
    # kernel correctness verification failure
    if success == False:
        avg_latency = 10000
    return kernel, avg_latency

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