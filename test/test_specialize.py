import torch
from sparta.specialization.specialize_kernel import specialize_matmul
from sparta.common.sparsity import TeSA, TesaAttr

def test_block_sparse_fp32():
    m, k, n = 1024, 1024, 1024
    in_tesa_attr = torch.Tensor(k, m)
    weight_tesa_attr = torch.Tensor(n, k)
    out_tesa_attr = torch.Tensor(n, m)

    in_tesa = TeSA(in_tesa_attr)
    weight_tesa = TeSA(weight_tesa_attr)
    out_tesa = TeSA(out_tesa_attr)

    block_size_m, block_size_k, block_size_n = 128, 8, 64
    in_tesa.set_transform_meta([block_size_k, block_size_m], 32)
    weight_tesa.set_transform_meta([block_size_n, block_size_k], 32)
    out_tesa.set_transform_meta([block_size_n, block_size_m], 32)

    latency, kernels, aggr_type = specialize_matmul((in_tesa, ), (weight_tesa, ), (out_tesa, ))

    print(f"overall latency is {latency}")

def test_block_sparse_quantize():
    m, k, n = 1024, 1024, 1024
    in_tesa_attr = torch.Tensor(k, m)
    weight_tesa_attr = torch.Tensor(n, k)
    out_tesa_attr = torch.Tensor(n, m)

    in_tesa = TeSA(in_tesa_attr)
    weight_tesa = TeSA(weight_tesa_attr)
    out_tesa = TeSA(out_tesa_attr)

    block_size_m, block_size_k, block_size_n = 128, 128, 64
    in_tesa.set_transform_meta([block_size_k, block_size_m], 8)
    weight_tesa.set_transform_meta([block_size_n, block_size_k], 8)
    out_tesa.set_transform_meta([block_size_n, block_size_m], 8)

    latency, kernels, aggr_type = specialize_matmul((in_tesa, ), (weight_tesa, ), (out_tesa, ))

    print(f"overall latency is {latency}")

def test_dense_sparse_quantize():
    m, k, n = 1024, 1024, 1024
    in_tesa_attr = torch.Tensor(k, m)
    weight_tesa_attr = torch.Tensor(n, k)
    out_tesa_attr = torch.Tensor(n, m)

    in_tesa = TeSA(in_tesa_attr)
    weight_tesa = TeSA(weight_tesa_attr)
    out_tesa = TeSA(out_tesa_attr)

    in_tesa.set_transform_meta(None, 8)
    weight_tesa.set_transform_meta(None, 8)
    out_tesa.set_transform_meta(None, 8)

    latency, kernels, aggr_type = specialize_matmul((in_tesa, ), (weight_tesa, ), (out_tesa, ))

    print(f"overall latency is {latency}")

test_block_sparse_fp32()
# test_dense_sparse_quantize()
# test_block_sparse_quantize()