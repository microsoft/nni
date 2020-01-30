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
        # index existing ones
        self.mask_index = [None for _ in range(num_dim)]

    def add_index_mask(self, dim, index):
        self.mask_index[dim] = index

    @staticmethod
    def merge_index(index_a, index_b):
        s = set()
        for num in index_a:
            s.add(num)
        for num in index_b:
            s.add(num)
        return torch.tensor(sorted(s))

    def merge(self, cmask):
        assert isinstance(cmask, CoarseMask)
        assert len(self.mask_index) == len(cmask.mask_index)
        for i, index in enumerate(self.mask_index):
            if index is None:
                self.mask_index[i] = cmask.mask_index[i]
            elif cmask.mask_index[i] is not None:
                self.mask_index[i] = CoarseMask.merge_index(self.mask_index[i],
                                                            cmask.mask_index[i])
        return self.mask_index

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

infer_from_inshape = {
    'ReLU': lambda module_masks, mask: relu_inshape(module_masks, mask),
    'aten::relu': lambda module_masks, mask: relu_inshape(module_masks, mask),
    'Conv2d': lambda module_masks, mask: conv2d_inshape(module_masks, mask),
    'MaxPool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::max_pool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::avg_pool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'AvgPool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::size': lambda module_masks, mask: size_inshape(module_masks, mask),
    'aten::view': lambda module_masks, mask: view_inshape(module_masks, mask),
    'Linear': lambda module_masks, mask: linear_inshape(module_masks, mask)
}

infer_from_outshape = {
    'Conv2d': lambda module_masks, mask: conv2d_outshape(module_masks, mask)
}

def linear_inshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[0] is None
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    return None

def view_inshape(module_masks, mask):
    """
    """
    # TODO: currently hard code view(N, -1)
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    output_cmask = CoarseMask(num_dim=2)
    # TODO: hard code for this case, %x : Float(64, 512, 1, 1)
    index = []
    for loc in mask.mask_index[1]:
        index.append(loc * 1)
    output_cmask.mask_index[1] = torch.tensor(index)
    module_masks.set_output_mask(output_cmask)
    return output_cmask


def size_inshape(module_masks, mask):
    """
    """
    return None

def maxpool2d_inshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    module_masks.set_output_mask(mask)
    return mask

def relu_inshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    # TODO: double check this assert, is it possible that a module is passed twice
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    module_masks.set_output_mask(mask)
    return mask # return shape of output tensor

def batchnorm2d_mask(module_masks, mask):
    """
    """
    assert 'weight' in mask and 'bias' in mask
    sum_mask = mask['weight'] + mask['bias']
    nonzero_index = torch.nonzero(sum_mask, as_tuple=True)[0]
    # infer shape of parameters
    param_cmask = CoarseMask(num_dim=1)
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
    def convert_to_coarse_mask(mask):
        assert 'weight' in mask
        assert isinstance(mask['weight'], torch.Tensor)
        cmask = None
        weight_mask = mask['weight']
        shape = weight_mask.size()
        ones = torch.ones(shape[1:])
        zeros = torch.zeros(shape[1:])
        index = []
        for i in range(shape[0]):
            if torch.all(torch.eq(weight_mask[i], ones)):
                index.append(i)
            elif torch.all(torch.eq(weight_mask[i], zeros)):
                continue
            else:
                index = None
                break
        if index is None:
            return None, None, None
        else:
            index = torch.LongTensor(index)
            weight_cmask = CoarseMask(num_dim=4)
            weight_cmask.add_index_mask(dim=0, index=index)
            bias_cmask = None
            if 'bias' in mask:
                bias_index = torch.nonzero(mask['bias'], as_tuple=True)[0]
                assert torch.all(torch.eq(index, bias_index))
                bias_cmask = CoarseMask(num_dim=1)
                bias_cmask.add_index_mask(dim=0, index=bias_index)
            return index, weight_cmask, bias_cmask
    index, weight_cmask, bias_cmask = convert_to_coarse_mask(mask)
    if index is None:
        # TODO: fine grained mask speedup
        return None, None
    # deal with coarse grain mask
    if 'weight' in module_masks.param_masks:
        module_masks.param_masks['weight'].merge(weight_cmask)
        module_masks.param_masks['bias'].merge(bias_cmask)
    else:
        module_masks.set_param_masks('weight', weight_cmask)
        module_masks.set_param_masks('bias', bias_cmask)
    output_cmask = CoarseMask(num_dim=4)
    output_cmask.add_index_mask(dim=1, index=index)
    if module_masks.output_mask is None:
        module_masks.set_output_mask(output_cmask)
    else:
        module_masks.output_mask.merge(output_cmask)
    return None, module_masks.output_mask

def conv2d_inshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    return None

def conv2d_outshape(module_masks, mask):
    """
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None

    if module_masks.output_mask is not None:
        assert isinstance(module_masks.output_mask, CoarseMask)
        # set shape of output
        mask = module_masks.output_mask.merge(mask)
    else:
        module_masks.output_mask = mask
    # infer shape of parameters
    weight_cmask = CoarseMask(num_dim=4)
    weight_cmask.add_index_mask(dim=0, index=mask.mask_index[1])
    bias_cmask = CoarseMask(num_dim=1)
    bias_cmask.add_index_mask(dim=0, index=mask.mask_index[1])
    module_masks.set_param_masks('weight', weight_cmask)
    module_masks.set_param_masks('bias', bias_cmask)
    # input shape is not changed
    return None # return shape of input tensor
    