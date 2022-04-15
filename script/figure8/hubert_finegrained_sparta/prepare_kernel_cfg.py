import os
import torch
import json
import sys
sys.path.append('../../../sparta/specialization/Template/finegrained_sparse_template/')
from spmm_finegrained_codegen import emit_finegrained_sparse_kernel_entry


hubert_ckpt_path = "artifact_hubert_finegrained_onnx_with_tesa"

def load_tesa(in_dir):

    data = {}

    tesa_path = os.path.join(in_dir, 'tesa')
    name_map_path = os.path.join(in_dir, 'tesaid_2_names')
    shape_path = os.path.join(in_dir, 'shape.json')
    state =torch.load(os.path.join(in_dir, 'state_dict.pth'))
    tesa = torch.load(tesa_path)
    name_map = torch.load(name_map_path)

    with open(shape_path, 'r') as f:
        shape = json.load(f)

    for tesaid in tesa:
        name = name_map[tesaid][0]
        # import ipdb; ipdb.set_trace()
        data[tesaid] = {'tesa': tesa[tesaid], 'shape':shape[name], 'state':{'weight':state[name+'.weight'], 'bias':state[name+'.bias']}}

    return data


def get_m2_k2_n2_from_m_k_n(m, k, n, HARDWARE='2080Ti'): # tuned results
    if HARDWARE == '2080Ti':
        if m==3072 and k==768 and n==4096:
            return 64, 16, 128
        elif m==768 and k==768 and n==4096:
            return 64, 1, 128
        elif m==768 and k==3072 and n==4096:
            return 32, 1, 256
        elif m==3072 and k==768 and n==1568:
            return 32, 8, 256
        elif m==768 and k==768 and n==1568:
            return 64, 1, 128
        elif m==768 and k==3072 and n==1568:
            return 32, 2, 128
        elif m==256 and k==768 and n==1568:
            return 32, 8, 32
        elif m==768 and k==512 and n==1568:
            return 32, 16, 128
    elif HARDWARE == 'MI50':
        if m==3072 and k==768 and n==4096:
            return 16, 32, 256
        elif m==768 and k==768 and n==4096:
            return 16, 32, 256
        elif m==768 and k==3072 and n==4096:
            return 16, 32, 256
        elif m==3072 and k==768 and n==1568:
            return 16, 32, 256
        elif m==768 and k==768 and n==1568:
            return 16, 32, 256
        elif m==768 and k==3072 and n==1568:
            return 16, 32, 256
        elif m==256 and k==768 and n==1568:
            return 16, 32, 256
        elif m==768 and k==512 and n==1568:
            return 16, 16, 256

    return 64, 8, 64


def codegen(ops, json_path, hardware='2080Ti'):
    sparta_kernels = []

    for tesa_id in ops:
        # if tesa_id <73:
        #     continue
        # print(tesa_id)
        op = ops[tesa_id]
        # print((op["shape"]["in_shape"][0]))
        if len(op["shape"]["in_shape"][0]) < 4:
            continue
        tesa_matrix = torch.reshape(op["tesa"]["weight"], (-1, )).detach().numpy()
        sp_matrix = torch.reshape(op["state"]["weight"], (-1, )).detach().numpy()
        TESAID = tesa_id
        
        N = int(op["shape"]["in_shape"][0][0])
        if len(op["shape"]["in_shape"][0]) == 4:
            N = N * int(op["shape"]["in_shape"][0][1])
        K = int(op["shape"]["in_shape"][0][2])
        M = int(op["shape"]["out_shape"][0][2])

        M_tile, K_tile, N_tile = get_m2_k2_n2_from_m_k_n(M, K, N, hardware)
        
        print(tesa_id, M, K, N, M_tile, K_tile, N_tile)

        dot_kernel_entry = emit_finegrained_sparse_kernel_entry(tesa_matrix, sp_matrix, M, N, K, M_tile, N_tile, K_tile, TESAID, False, False)
        # fused_dot_add_kernel_entry = emit_finegrained_sparse_kernel_entry(tesa_matrix, sp_matrix, M, N, K, M_tile, N_tile, K_tile, TESAID, False, True)

        sparta_kernels.append(dot_kernel_entry)
        # sparta_kernels.append(fused_dot_add_kernel_entry)

    with open(json_path, "w") as fout:
        json.dump(sparta_kernels, fout)


hubert_ops = load_tesa(hubert_ckpt_path)

codegen(hubert_ops, "hubert_sparta_finegrained_kernels.json")

os.system("rm ~/.cache/nnfusion/kernel_cache.db")
nnfusion_home = os.getenv('NNFUSION_HOME')
assert nnfusion_home is not None
os.system(f"python {nnfusion_home}/src/tools/nnfusion/kernel_db/convert_external_sparta_tesa.py hubert_sparta_finegrained_kernels.json")
os.system("cp ~/.cache/nnfusion/kernel_cache.db ./")