from audioop import avg
import torch
import random
from numpy import array
import subprocess
import json

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
    non_zero_block_num = 0
    for _i in range(size_h//block_h):
        rows.append(len(cols))
        for _j in range(size_w//block_w):
            i_start = _i * block_h
            i_end = (_i+1) * block_h
            j_start = _j * block_w
            j_end = (_j+1) * block_w
            if torch.sum(m_tensor[i_start:i_end, j_start:j_end]) > 0:
                non_zero_block_num += 1
                cols.append(_j)
                values.extend(v_tensor[i_start:i_end,j_start:j_end].flatten().tolist())
    print(f"non_zero_block_num: {non_zero_block_num}")
    rows.append(len(cols))
    return rows, cols, values

def bcsr_generate(sparsity):
    k, n = 1024, 1024
    block_size = 32

    block_num_k = int(k / block_size)
    block_num_n = int(n / block_size)

    block_num = int(block_num_k * block_num_n)
    nonzero_block_num = int(block_num * (1-sparsity))

    block_ids = [i for i in range(block_num)]
    nonzero_block_ids = random.sample(block_ids, nonzero_block_num)

    tesa=torch.zeros(1024,1024)

    for block_k_idx in range(block_num_k):
        for block_n_idx in range(block_num_n):
            cur_block_idx = block_k_idx * block_num_n + block_n_idx
            matrix_k_idx_start = block_k_idx * block_size
            matrix_k_idx_end = (block_k_idx+1) * block_size
            matrix_n_idx_start = block_n_idx * block_size
            matrix_n_idx_end = (block_n_idx+1) * block_size
            if cur_block_idx in nonzero_block_ids:
                tesa[matrix_k_idx_start:matrix_k_idx_end,matrix_n_idx_start:matrix_n_idx_end] = 1
    
    print("Block sparsity generated")
    
    rows, cols, vals = convert_to_block_csr(tesa, tesa, 16, 32)
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    
    write_array(rows, rows_path)
    write_array(cols, cols_path)
    write_array(vals, vals_path)

def bcsr_generate_int8(sparsity):
    k, n = 1024, 1024
    block_size = 32

    block_num_k = int(k / block_size)
    block_num_n = int(n / block_size)

    block_num = int(block_num_k * block_num_n)
    nonzero_block_num = int(block_num * (1-sparsity))

    block_ids = [i for i in range(block_num)]
    nonzero_block_ids = random.sample(block_ids, nonzero_block_num)

    tesa=torch.zeros(1024,1024)

    for block_k_idx in range(block_num_k):
        for block_n_idx in range(block_num_n):
            cur_block_idx = block_k_idx * block_num_n + block_n_idx
            matrix_k_idx_start = block_k_idx * block_size
            matrix_k_idx_end = (block_k_idx+1) * block_size
            matrix_n_idx_start = block_n_idx * block_size
            matrix_n_idx_end = (block_n_idx+1) * block_size
            if cur_block_idx in nonzero_block_ids:
                tesa[matrix_k_idx_start:matrix_k_idx_end,matrix_n_idx_start:matrix_n_idx_end] = 1
    
    print("Block sparsity generated")
    
    rows, cols, vals = convert_to_block_csr(tesa, tesa, 32, 32)
    rows_path = f"bcsr_row.bin"
    cols_path = f"bcsr_col.bin"
    vals_path = f"bcsr_val.bin"
    
    write_array(rows, rows_path)
    write_array(cols, cols_path)
    write_array(vals, vals_path)

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

def run_openai_kernel():
    compile_cmd = "nvcc -gencode arch=compute_75,code=sm_75 -o openai_blocksparse openai_blocksparse.cu"
    output_file_name = f"output_log.txt"
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=6000)
    latencys = []
    for i in range(2):
        command = f'./openai_blocksparse > {output_file_name}'
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=6000)

        if i == 0:
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_name)))
    avg_latency = sum(latencys) / len(latencys)
    return avg_latency

def run_sparta_kernel():
    compile_cmd = "nvcc -gencode arch=compute_75,code=sm_75 -o sparta_kernel sparta_kernel.cu"
    output_file_name = f"output_log.txt"
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=6000)
    latencys = []
    for i in range(2):
        command = f'./sparta_kernel > {output_file_name}'
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=6000)

        if i == 0:
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_name)))
    avg_latency = sum(latencys) / len(latencys)
    return avg_latency

def run_sparta_kernel_int8():
    compile_cmd = "nvcc -gencode arch=compute_75,code=sm_75 -o sparta_kernel_int8 sparta_kernel_int8.cu"
    output_file_name = f"output_log.txt"
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=6000)
    latencys = []
    for i in range(2):
        command = f'./sparta_kernel_int8 > {output_file_name}'
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=6000)

        if i == 0:
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_name)))
    avg_latency = sum(latencys) / len(latencys)
    return avg_latency

def run_sputnik_kernel(sparsity):
    compile_cmd = 'nvcc -forward-unknown-to-host-compiler -I/usr/local/cuda/include -I/root/sputnik  -L/usr/local/cuda/lib64  -L/usr/local/lib -lcudart -lspmm  --generate-code=arch=compute_75,code=sm_75 -std=c++14  sputnik.cu -o sputnik'
    output_file_name = f"output_log.txt"
    subprocess.check_output(compile_cmd, shell = True, universal_newlines=True, timeout=6000)
    latencys = []
    for i in range(2):
        command = f'./sputnik {sparsity}> {output_file_name}'
        #os.system('nvprof --unified-memory-profiling off ./{} 2> a_{}.txt'.format(Path(file_name).stem, file_name))
        #os.system(command)
        subprocess.check_output(command, shell = True, universal_newlines=True, timeout=6000)

        if i == 0:
            # warmup
            continue
        latencys.append(get_kernel_run_time('{}'.format(output_file_name)))
    avg_latency = sum(latencys) / len(latencys)
    return avg_latency
def main():
    sparsity_list = [0.7, 0.8, 0.9]
    result = {'sparta': [], 'openai': [], 'sparta_int8': [], 'sputnik':[]}
    for sparsity in sparsity_list:
        bcsr_generate(sparsity)
        openai_latency = run_openai_kernel()
        sparta_latency = run_sparta_kernel()
        spunik_latency = run_sputnik_kernel(sparsity)
        bcsr_generate_int8(sparsity)
        sparta_int8_latency = run_sparta_kernel_int8()
        result['openai'].append(openai_latency)
        result['sparta'].append(sparta_latency)
        result['sparta_int8'].append(sparta_int8_latency)
        result['sputnik'].append(spunik_latency)
    with open('raw_data.json', 'w') as f:
        json.dump(result, f)

main()
#bcsr_generate_int8(0.9)