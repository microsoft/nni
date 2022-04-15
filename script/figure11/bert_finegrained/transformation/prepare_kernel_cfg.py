from sparta.common.utils import generate_sputnik_sparse_cfg, inject_kernel
import argparse
import os
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', help='input dir')
    parser.add_argument('--out_dir', help='output_dir')
    args = parser.parse_args()
    tesa_path = os.path.join(args.in_dir, 'tesa')
    state_path = os.path.join(args.in_dir, 'state_dict.pth')
    id_map_path = os.path.join(args.in_dir, 'tesaid_2_names')
    id_map = os.path.join(args.in_dir, 'tesaid_2_names')

    id_maps = torch.load(id_map_path)
    name_2_tid= {}
    id_2_name = {}
    for tid, names in id_maps.items():
        id_2_name[tid] = names[0]
        name_2_tid[names[0]] =tid

    generate_sputnik_sparse_cfg(tesa_path, state_path, id_map, args.out_dir)
    onnx_path = os.path.join(args.in_dir, 'model_tesa.onnx')
    os.system('cp {} {}'.format(onnx_path, args.out_dir))