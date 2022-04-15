import torch
import torch.nn as nn

from sparta.common import SparseModuleInfo, ModelSparsityInfo
from sparta.specialization import specialize_kernel
from sparta.transformation.transform_policy import TransformPolicy

__all__ = ['optimize_and_rebuild']

def rebuild_sparse_model_pytorch(model: nn.Module, opt_modules: dict):
    """
    Important: users can directly use our optimized module,
    and also they can use the meta/wrapper module, which will be replace
    with the optimized module here.
    """
    # first support pytorch
    # return mocked for now
    return model

def rebuild_sparse_model_nnfusion(model: nn.Module, opt_modules: dict):
    ...

def is_fast_pass(post_sparsity: ModelSparsityInfo) -> bool:
    ...

def fast_pass_for_dense(model: nn.Module, backend = 'pytorch', device_info = None):
    ...

def optimize_and_rebuild(model: nn.Module,
                         post_sparsity: ModelSparsityInfo,
                         backend = 'pytorch',
                         device_info = None):
    # fast pass when the model is dense.
    # in some cases, the sparse model is directly converted
    # to a small dense in the previous stage
    if is_fast_pass(post_sparsity):
        opt_model = fast_pass_for_dense(model, backend, device_info)
        return opt_model

    # the pass for sparse operator optimizations
    opt_modules = {}
    # init a transformation policy
    tpolicy = TransformPolicy(device_info)
    for module_name, module_sparsity in post_sparsity.items():
        transformed = tpolicy.transform_module(module_sparsity)
        opt_modules[module_name] = transformed
    # rebuild the sparse model: module replacement or nnfusion
    if backend == 'pytorch':
        opt_model = rebuild_sparse_model_pytorch(model, opt_modules)
    elif backend == 'nnfusion':
        opt_model = rebuild_sparse_model_nnfusion(model, opt_modules)
    else:
        raise
    return opt_model