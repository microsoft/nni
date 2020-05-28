# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
For each operation or module, there are two functions.
One is given output shape, infer its input shape and initialization parameters (e.g., weight's shape)
The other is given input shape, infer its output shape and initialization parameters (e.g., weight's shape)
"""

import torch

class CoarseMask:
    """
    Coarse grained mask for a given tensor, here tensor could be weights,
    input tensor, or output tensor
    """
    def __init__(self, num_dim):
        """
        Parameters
        ----------
        num_dim : int
            The number of dimensions of the tensor that will be masked
        """
        self.mask_index = [None for _ in range(num_dim)]

    def add_index_mask(self, dim, index):
        """
        Add mask for the specified dimension

        Parameters
        ----------
        dim : int
            The dimension to add mask
        index : tensor
            The mask for this dimension, its a 1 dimension tensor which specifies
            the index of the elements that are not pruned
        """
        self.mask_index[dim] = index

    @staticmethod
    def merge_index(index_a, index_b):
        """
        Parameters
        ----------
        index_a : tensor
            One index (1-dimension) tensor
        index_b : tensor
            The other index (1-dimension) tensor

        Returns
        -------
        tensor
            The merged index (1-dimension) tensor
        """
        s = set()
        for num in index_a:
            s.add(num)
        for num in index_b:
            s.add(num)
        return torch.tensor(sorted(s)) # pylint: disable=not-callable

    def merge(self, cmask):
        """
        Merge another CoarseMask

        Parameters
        ----------
        cmask : CoarseMask
            Another CoarseMask to merge

        Returns
        -------
        list
            The member variable ```mask_index```
        """
        assert isinstance(cmask, CoarseMask)
        assert len(self.mask_index) == len(cmask.mask_index), \
            "Only masks with the same number of dimensions can be merged"
        for i, index in enumerate(self.mask_index):
            if index is None:
                self.mask_index[i] = cmask.mask_index[i]
            elif cmask.mask_index[i] is not None:
                self.mask_index[i] = CoarseMask.merge_index(self.mask_index[i],
                                                            cmask.mask_index[i])
        return self.mask_index

    def __repr__(self):
        return 'mask_index: {}'.format(self.mask_index)

class ModuleMasks:
    """
    The masks of a module, including the masks for weights, inputs, output
    """
    def __init__(self, module_name):
        """
        Parameters
        ----------
        module_name : str
            The name of the module or function
        """
        self.module_name = module_name
        self.param_masks = dict()
        self.input_mask = None
        self.output_mask = None

    def set_param_masks(self, name, mask):
        """
        Parameters
        ----------
        name : str
            The name of the weight
        mask : CoarseMask
            The mask for this weight
        """
        self.param_masks[name] = mask

    def set_input_mask(self, mask):
        """
        Parameters
        ----------
        mask : CoarseMask
            The mask for input
        """
        self.input_mask = mask

    def set_output_mask(self, mask):
        """
        Parameters
        ----------
        mask : CoarseMask
            The mask for output
        """
        self.output_mask = mask

    def __repr__(self):
        return 'input_mask: {}, output_mask: {}, param_masks: {}'.format(
            self.input_mask, self.output_mask, self.param_masks
        )

"""
Infer input and output shape of a module/function from its weight mask
"""
infer_from_mask = {
    'BatchNorm2d': lambda module_masks, mask: batchnorm2d_mask(module_masks, mask),
    'Conv2d': lambda module_masks, mask: conv2d_mask(module_masks, mask)
}

"""
Infer output and weight shape of a module/function from its input shape
"""
infer_from_inshape = {
    'ReLU': lambda module_masks, mask: relu_inshape(module_masks, mask),
    'aten::relu': lambda module_masks, mask: relu_inshape(module_masks, mask),
    'Conv2d': lambda module_masks, mask: conv2d_inshape(module_masks, mask),
    'MaxPool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::max_pool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::avg_pool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'AvgPool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'AdaptiveAvgPool2d': lambda module_masks, mask: maxpool2d_inshape(module_masks, mask),
    'aten::size': lambda module_masks, mask: size_inshape(module_masks, mask),
    'aten::view': lambda module_masks, mask, shape: view_inshape(module_masks, mask, shape),
    'aten::flatten': lambda module_masks, mask, shape: view_inshape(module_masks, mask, shape), # support only start_dim=1
    'Linear': lambda module_masks, mask: linear_inshape(module_masks, mask),
    'BatchNorm2d': lambda module_masks, mask: batchnorm2d_inshape(module_masks, mask)
}

"""
Infer input and weight shape of a module/function from its output shape
"""
infer_from_outshape = {
    'Conv2d': lambda module_masks, mask: conv2d_outshape(module_masks, mask)
}

def batchnorm2d_inshape(module_masks, mask):
    """
    We assume only the second dimension has coarse grained mask

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the batchnorm2d
    mask : CoarseMask
        The mask of its input tensor

    Returns
    -------
    CoarseMask
        The mask of its output tensor
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None
    module_masks.set_input_mask(mask)
    module_masks.set_output_mask(mask)
    weight_cmask = CoarseMask(num_dim=1)
    weight_cmask.add_index_mask(dim=0, index=mask.mask_index[1])
    module_masks.set_param_masks('weight', weight_cmask)
    module_masks.set_param_masks('bias', weight_cmask)
    return mask

def linear_inshape(module_masks, mask):
    """
    Coarse grained input mask does not change the shape of weights and output tensor

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the linear
    mask : CoarseMask
        The mask of its input tensor

    Returns
    -------
    CoarseMask
        The mask of its output tensor, ```None``` means shape of output tensor is not changed
    """
    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[0] is None
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    return None

def view_inshape(module_masks, mask, shape):
    """
    This is a limited support

    TODO: consider replace tensor.view with nn.Flatten, because tensor.view is not
    included in module, thus, cannot be replaced by our framework.

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the ```view``` op
    mask : CoarseMask
        The mask of its input tensor
    shape : dict
        Original shape of its input and output tensors

    Returns
    -------
    CoarseMask
        The mask of its output tensor
    """
    # NOTE: the case constrained by the following four asserts
    assert shape['in_shape'][0] == shape['out_shape'][0]
    assert len(shape['in_shape']) == 4
    assert len(shape['out_shape']) == 2
    assert shape['out_shape'][1] == shape['in_shape'][1]*shape['in_shape'][2]*shape['in_shape'][3]

    assert isinstance(mask, CoarseMask)
    assert mask.mask_index[1] is not None
    assert mask.mask_index[0] is None
    assert mask.mask_index[2] is None
    assert mask.mask_index[3] is None
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    output_cmask = CoarseMask(num_dim=2)
    index = []
    step_size = shape['in_shape'][2] * shape['in_shape'][3]
    for loc in mask.mask_index[1]:
        index.extend([loc * step_size + i for i in range(step_size)])
    output_cmask.add_index_mask(dim=1, index=torch.tensor(index)) # pylint: disable=not-callable
    module_masks.set_output_mask(output_cmask)
    return output_cmask


def size_inshape(module_masks, mask):
    """
    No need to do anything for this ```size``` op
    """
    return None

def maxpool2d_inshape(module_masks, mask):
    """
    Assume only the second dimension is masked

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the maxpool2d
    mask : CoarseMask
        The mask of its input tensor

    Returns
    -------
    CoarseMask
        The mask of its output tensor
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
    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the relu
    mask : CoarseMask
        The mask of its input tensor

    Returns
    -------
    CoarseMask
        The mask of its output tensor
    """
    assert isinstance(mask, CoarseMask)
    # TODO: double check this assert, is it possible that a module is passed twice
    assert module_masks.input_mask is None, "A relu op can only be processed once"
    module_masks.set_input_mask(mask)
    module_masks.set_output_mask(mask)
    return mask

def batchnorm2d_mask(module_masks, mask):
    """
    Infer input and output shape from weight mask

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the batchnorm2d
    mask : dict
        The mask of its weights, from the user provided mask file

    Returns
    -------
    CoarseMask, CoarseMask
        The mask of its input tensor, the mask of its output tensor
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
    Infer input and output shape from weight mask

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the conv2d
    mask : dict
        The mask of its weights, from the user provided mask file

    Returns
    -------
    CoarseMask, CoarseMask
        The mask of its input tensor, the mask of its output tensor
    """
    def convert_to_coarse_mask(mask):
        """
        Parameters
        ----------
        mask : dict
            Weight mask from user provided mask file

        Returns
        -------
        LongTensor, CoarseMask, CoarseMask
            Index of the masked dimension, weight mask, bias mask
        """
        assert 'weight' in mask
        assert isinstance(mask['weight'], torch.Tensor)
        weight_mask = mask['weight']
        shape = weight_mask.size()
        ones = torch.ones(shape[1:]).to(weight_mask.device)
        zeros = torch.zeros(shape[1:]).to(weight_mask.device)
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
            index = torch.LongTensor(index).to(weight_mask.device)
            weight_cmask = CoarseMask(num_dim=4)
            weight_cmask.add_index_mask(dim=0, index=index)
            bias_cmask = None
            if 'bias' in mask and mask['bias'] is not None:
                bias_index = torch.nonzero(mask['bias'], as_tuple=True)[0]
                assert torch.all(torch.eq(index, bias_index)), \
                    "bias mask should be consistent with weight mask"
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
    Shape change of input tensor does not affect the shape of its output tensor

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the conv2d
    mask : CoarseMask
        The mask of its input tensor

    Returns
    -------
    CoarseMask
        The mask of its output tensor
    """
    assert isinstance(mask, CoarseMask)
    assert module_masks.input_mask is None
    module_masks.set_input_mask(mask)
    return None

def conv2d_outshape(module_masks, mask):
    """
    Assume only the second dimension is masked

    Parameters
    ----------
    module_masks : ModuleMasks
        The ModuleMasks instance of the conv2d
    mask : CoarseMask
        The mask of its output tensor

    Returns
    -------
    CoarseMask
        The mask of its input tensor
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
    return None
    