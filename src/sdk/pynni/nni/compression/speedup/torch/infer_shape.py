# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
For each operation or module, there are two functions.
One is given output shape, infer its input shape and initialization parameters (e.g., weight's shape)
The other is given input shape, infer its output shape and initialization parameters (e.g., weight's shape)
"""

import torch

class CoarseMask:
    def __init__(self, num_dim):
        self.mask_index = [None for _ in range(num_dim)]

    def add_index_mask(self, dim, index):
        self.mask_index[dim] = index

class ModuleMasks:
    def __init__(self, module_name):
        """
        """
        self.module_name = module_name
        self.param_masks = dict()
        self.input_mask = None
        self.output_mask = None
    
    def set_param_masks(self, name, mask):
        self.param_masks[name] = mask

    def set_input_mask(self, mask):
        self.input_mask = mask

    def set_output_mask(self, mask):
        self.output_mask = mask


infer_from_mask = {
    'BatchNorm2d': lambda module_masks, mask: batchnorm2d_mask(module_masks, mask),
    'Conv2d': lambda module_masks, mask: conv2d_mask(module_masks, mask)
}

infer_from_inshape = {}

infer_from_outshape = {
    'Conv2d': lambda module_masks, mask: conv2d_outshape(module_masks, mask)
}

def batchnorm2d_mask(module_masks, mask):
    """
    """
    assert 'weight' in mask and 'bias' in mask
    sum_mask = mask['weight'] + mask['bias']
    nonzero_index = torch.nonzero(sum_mask, as_tuple=True)[0]
    # infer shape of parameters
    param_cmask = CoarseMask(num_dim=0)
    param_cmask.add_index_mask(dim=0, index=nonzero_index)
    module_masks.set_param_masks('weight', param_cmask)
    module_masks.set_param_masks('bias', param_cmask)
    # infer shape of input tensor
    input_cmask = CoarseMask(num_dim=4)
    input_cmask.add_index_mask(dim=1,
                               index=torch.nonzero(mask['weight'], as_tuple=True)[0])
    module_masks.set_input_mask(input_cmask)
    # infer shape of output tensor
    output_cmask = CoarseMask(num_dim=4)
    output_cmask.add_index_mask(dim=1, index=nonzero_index)
    module_masks.set_output_mask(output_cmask)
    return input_cmask, output_cmask

def conv2d_mask(module_masks, mask):
    """
    """

def conv2d_outshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None
    if module_masks.output_mask is not None:
        # ...
        return
    #...
    