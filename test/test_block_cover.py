import random
import torch
import torch.nn as nn
from sparta.common import SparseModuleInfo, TeSA
from sparta.transformation import TransformPolicy

def sparsify_tensor_unstructured(tensor: torch.Tensor, sparsity_ratio: float = 0.5):
    (d1, d2) = tensor.size()
    for i in range(d1):
        for j in range(d2):
            if random.random() > sparsity_ratio:
                tensor[i][j] = 32
    return tensor

def specify_precision(tensor: torch.Tensor, bits: int):
    (d1, d2) = tensor.size()
    for i in range(d1):
        for j in range(d2):
            tensor[i][j] = bits
    return tensor

# prepare sparse module
# (32,1024) * (1024, 1024) -> (32, 1024)
linear = nn.Linear(1024, 1024)
weight_mask = torch.zeros_like(linear.weight, dtype=torch.int8)
weight_mask = sparsify_tensor_unstructured(weight_mask, 0.9)
input_mask = specify_precision(torch.zeros(32, 1024), 32)
output_mask = specify_precision(torch.zeros(32, 1024), 32)
sparse_linear = SparseModuleInfo(
    'test_linear',
    linear,
    weight_tesa=weight_mask,
    input_tesa=input_mask,
    output_tesa=output_mask)

tp = TransformPolicy()
print('start transforming sparse linear...')
tp.transform_module(sparse_linear)
print('transformation done.')