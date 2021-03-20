# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def cat_conflict(node, input_masks, output_masks):
    # TODO does cat need to resolve the mask conflict?
    pass

def add_conflict_unmask(node, input_masks, output_mask):
    """
    Find the positions of the input masks that need
    to be unmasked to resolve the mask conflicts.

    Parameters
    ----------
    node: NodePyGroup
        The add operation node that to resolve conflict.
    input_masks: list
        List of tensors, each element corresponds to a mask tensor of inputs.
    output_mask: torch.Tensor
        The mask of the output tensor.
    Returns
    ------
    need_unmask: list
        This list has the same length with input masks. The element of the list
        will be None or torch.Tensor, if it is None, then the corresponding input
        mask doesn't need to unmask any value, else we should unmask the values in the
        tensor.
    """
    
    # in the add operation, we should align the input mask
    # with the output mask.
    assert isinstance(input_masks, list)
    assert isinstance(output_mask, torch.Tensor)
    need_unmask = []
    for t_in in input_masks:
        # find the position that was masked(0) in the input tensor
        # but not masked in the output tensor(1)
        need_unmask.append(output_mask - t_in)
    for i, t_unmask in enumerate(need_unmask):
        if torch.sum(t_unmask) == 0:
            # no need to unmask any value
            need_unmask[i] = None
    return need_unmask

def add_conflict_padding(node, input_masks, output_mask):
    """
    Return the reindex tensor of each input. This function should only
    be called while using structure pruning.

    Parameters
    ----------
    node: NodePyGroup
        The add operation node that to resolve conflict.
    input_masks: list
        List of tensors, each element corresponds to a mask tensor of inputs.
    output_mask: torch.Tensor
        The mask of the output tensor.
    Returns
    ------
    need_unmask: list
        This list has the same length with input masks. The element of the list
        will be None or torch.Tensor, if it is None, then the corresponding input
        mask doesn't need to unmask any value, else we should unmask the values in the
        tensor.
    """
    
    # in the add operation, we should align the input mask
    # with the output mask.
    assert isinstance(input_masks, list)
    assert isinstance(output_mask, torch.Tensor)
    # reduce the mask in channel-dimension
    shape = list(output_mask.size())
    dim_list = list(range(len(shape)))
    dim_list.remove(1)
    out_c_sum = torch.sum(output_mask, dim_list)
    input_c_sum = [torch.sum(x, dim_list) for x in input_masks]
    # c_out_prund = out_c_sum == 0
    c_out_remained = out_c_sum > 0
    reindexes = []
    for c_in_mask in input_c_sum:
        c_pruned = c_in_mask == 0
        reindexes.append(c_pruned[c_out_remained])
    return reindexes


ConflictUnmask = {
    'aten::cat': cat_conflict,
    'aten::add': add_conflict_unmask,
    'aten::add_': add_conflict_unmask,
    'aten::mul': add_conflict_unmask,
    'aten::mul_': add_conflict_unmask

}

ConflictPadding = {
    'aten::cat': cat_conflict,
    'aten::add': add_conflict_padding,
    'aten::add_': add_conflict_padding,
    'aten::mul': add_conflict_padding,
    'aten::mul_': add_conflict_padding

}

def calc_unmask(node, input_masks, output_mask):
    cacl_func = ConflictUnmask[node.op_type]
    return cacl_func(node, input_masks, output_mask)

def calc_padding(node, input_masks, output_mask):
    calc_func = ConflictPadding[node.op_type]
    return calc_func(node, input_masks, output_mask)