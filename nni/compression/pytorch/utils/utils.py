# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch

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

def randomize_tensor(tensor, start=1, end=10):
    """
    Randomize the target tensor according to the given
    range.
    """
    assert isinstance(tensor, torch.Tensor)
    if tensor.dtype in torch_integer_dtype:
        # integer tensor can only be randomized by the torch.randint
        torch.randint(start, end, tensor.size(), out=tensor.data, dtype=tensor.dtype)
    else:
        # we can use nn.init.uniform_ to randomize this tensor
        # Note: the tensor that with integer type cannot be randomize
        # with nn.init.uniform_
        torch.nn.init.uniform_(tensor.data, start, end)
    