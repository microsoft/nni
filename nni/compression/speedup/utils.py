# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
import logging
from typing import Any, Dict, List, Tuple

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten, _register_pytree_node, Context


torch_float_dtype = [torch.float, torch.float16,
                     torch.float32, torch.float64, torch.half, torch.double]

torch_integer_dtype = [torch.uint8, torch.int16,
                       torch.short, torch.int32, torch.long, torch.bool]


def _idict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _idict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return immutable_dict((key, value) for key, value in zip(context, values))

def _ilist_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None

def _ilist_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return immutable_list(values)

_register_pytree_node(immutable_dict, _idict_flatten, _idict_unflatten)
_register_pytree_node(immutable_list, _ilist_flatten, _ilist_unflatten)


def randomize_tensor_inplace(tensor: torch.Tensor, start=None, end=None):
    """
    Randomize the target tensor according to the given range.
    """
    assert isinstance(tensor, torch.Tensor)
    if start is None and end is None:
        start, end = tensor.min(), tensor.max()
    assert start is not None and end is not None
    if tensor.dtype in torch_integer_dtype:
        # integer tensor can only be randomized by the torch.randint
        torch.randint(int(start), int(end + 1), tensor.size(),
                      out=tensor.data, dtype=tensor.dtype)
    else:
        # we can use nn.init.uniform_ to randomize this tensor
        # Note: the tensor that with integer type cannot be randomize
        # with nn.init.uniform_
        torch.nn.init.uniform_(tensor.data, start, end)


def randomize_if_tensor(obj, batch_dim, batch_size):
    if isinstance(obj, torch.Tensor):
        new_obj = obj.clone().detach().contiguous()
        if obj.numel() != 1 and obj.dim() > batch_dim and obj.size(batch_dim) == batch_size:
            if new_obj.dtype in torch_float_dtype:
                # NOTE: this is a workaround to avoid relu/relu6 ...
                randomize_tensor_inplace(new_obj, 0.1, 8.0)
            else:
                randomize_tensor_inplace(new_obj)
        return new_obj
    else:
        try:
            return deepcopy(obj)
        except Exception:
            return obj


def randomize_like_with_shape(shape, ori_t):
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

    if dtype in torch_integer_dtype:
        return torch.randint(lower_bound.long(), higher_bound.long() + 1, shape, dtype=dtype, device=device)
    else:
        return torch.rand(shape, dtype=dtype, device=device, requires_grad=require_grad)


def tree_map_zip(fn: Any, *pytrees):
    assert len(pytrees) > 0
    if len(pytrees) == 1:
        return tree_map(fn, pytrees[0])
    else:
        flat_args_list, spec_list = [], []
        for pytree in pytrees:
            flat_args, spec = tree_flatten(pytree)
            flat_args_list.append(flat_args)
            spec_list.append(spec)
        assert all(len(args) == len(flat_args_list[0]) for args in flat_args_list), 'Inconsistent tree nodes length.'
        return tree_unflatten([fn(*args) for args in zip(*flat_args_list)], spec_list[0])


def poss_deepcopy(o, logger: logging.Logger = None) -> Any:
    try:
        new_o = deepcopy(o)
    except Exception as e:
        if logger is not None:
            logger.warning(str(e))
        else:
            print(str(e))
        new_o = o
    return new_o
