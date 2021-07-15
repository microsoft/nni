# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from .shape_dependency import ReshapeDependency

torch_float_dtype = [torch.float, torch.float16, torch.float32, torch.float64, torch.half, torch.double]
torch_integer_dtype = [torch.uint8, torch.int16, torch.short, torch.int32, torch.long, torch.bool]

def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None


def rand_like_with_shape(shape, ori_t):
    """
    Return a new random tensor like the original
    tensor.
    """
    assert isinstance(ori_t, torch.Tensor)
    device = ori_t.device
    dtype = ori_t.dtype
    require_grad = ori_t.requires_grad
    lower_bound = torch.min(ori_t)
    higher_bound = torch.max(ori_t)
    if dtype in [torch.uint8, torch.int16, torch.short, torch.int16, torch.long, torch.bool]:
        return torch.randint(lower_bound, higher_bound+1, shape, dtype=dtype, device=device)
    else:
        return torch.rand(shape, dtype=dtype, device=device, requires_grad=require_grad)

def randomize_tensor(tensor, start=1, end=100):
    """
    Randomize the target tensor according to the given
    range.
    """
    assert isinstance(tensor, torch.Tensor)
    if tensor.dtype in torch_integer_dtype:
        # integer tensor can only be randomized by the torch.randint
        # torch.randint(int(start), int(end), tensor.size(), out=tensor.data, dtype=tensor.dtype)
        pass
    else:
        # we can use nn.init.uniform_ to randomize this tensor
        # Note: the tensor that with integer type cannot be randomize
        # with nn.init.uniform_
        torch.nn.init.uniform_(tensor.data, start, end)


def not_safe_to_prune(model, dummy_input):
    """
    Get the layers that are not safe to prune(may bring the shape conflict).
    For example, if the output tensor of a conv layer is directly followed by
    a shape-dependent function(such as reshape/view), then this conv layer
    may be not safe to be pruned. Pruning may change the output shape of
    this conv layer and result in shape problems. This function find all the
    layers that directly followed by the shape-dependent functions(view, reshape, etc).
    If you run the inference after the speedup and run into a shape related error,
    please exclude the layers returned by this function and try again.

    Parameters
    ----------
    model: torch.nn.Module
        The target model to prune.
    dummy_input: torch.Tensor/list of torch.Tensor/tuple of Tensor
    """
    reshape_dset = ReshapeDependency(model, dummy_input)
    return reshape_dset.dependency_sets