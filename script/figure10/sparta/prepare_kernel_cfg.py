from sparta.common.utils import generate_block_quantize_cfg, inject_kernel
import argparse
import os
import json
import torch
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', help='input dir')
    parser.add_argument('--out_dir', help='output_dir')
    args = parser.parse_args()
    tesa_path = os.path.join(args.in_dir, 'tesa')
    state_path = os.path.join(args.in_dir, 'state_dict.pth')
    id_map_path = os.path.join(args.in_dir, 'tesaid_2_names')
    id_map = os.path.join(args.in_dir, 'tesaid_2_names')
    sparse_block_cfg = {}
    with open('kernel_dict.json', 'r') as f:
        kernels = json.load(f)

    id_maps = torch.load(id_map_path)
    name_2_tid= {}
    id_2_name = {}
    for tid, names in id_maps.items():
        id_2_name[tid] = names[0]
        name_2_tid[names[0]] =tid

    for name in kernels:
        tid = name_2_tid[name]
        sparse_block_cfg[tid] = (kernels[name]['block_size_k'], kernels[name]['block_size_n'])
    generate_block_quantize_cfg(tesa_path, state_path, id_map, args.out_dir, sparse_block_cfg=sparse_block_cfg, sparsity_threshold=2)
    onnx_path = os.path.join(args.in_dir, 'model_tesa.onnx')
    os.system('cp {} {}'.format(onnx_path, args.out_dir))
    kernel_path = 'kernel_dict.json'
    template_path = 'block_sparse_template_bias_row.json'
    
    
    inject_kernel(template_path, kernel_path, 'BlockQuantizeDotAdd', id_map_path, os.path.join(args.out_dir, 'kernel'))
    # use fine-grained float32 kernel to run the coresponding parts
    lines = []
    fp32_mask = torch.load(os.path.join(args.in_dir, 'fp32_mask.pth'))
    cfg_path = os.path.join(args.out_dir, 'config')
    fp_tesaids = [name_2_tid[x] for x in fp32_mask]
    with open(cfg_path, 'r') as f:
        lines = f.readlines()
    with open(cfg_path, 'w') as f:
        for line in lines:
            tmp = re.split(' ', line)
            if int(tmp[0]) in fp_tesaids:
                f.write('{} Sputnik\n'.format(int(tmp[0])))
            else:
                f.write(line)
