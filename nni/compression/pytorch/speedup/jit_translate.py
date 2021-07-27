# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import logging
from functools import partial
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def translate_list(list_node, speedup=None):
    """
    Get the list of values from the list construct node.
    Parameters
    ---------
    list_node: Torch.C.Value
        The cpp node of the target list.
    speedup: ModuleSpeed
        The Module speedup module.
    Returns
    -------
    values: list
        The list of values in the target cpp list node.
    """
    # the node that create the list
    create_node = list_node.node()
    assert create_node.kind() == 'prim::ListConstruct'
    inputs = list(create_node.inputs())
    values = []
    for _i in inputs:
        debugName = _i.debugName()
        if speedup is not None and debugName in speedup.internal_result:
            # this value is the result of the other nodes, such as
            # ate::size
            values.append(speedup.internal_result[debugName].item())
        else:
            # if the corresponding value is a constant
            values.append(_i.toIValue())
    return values


def parse_constant(cvalue, speedup):
    """
    Parse the constant values from this Node
    Parameters
    ----------
    cvalue: Torch.C.Value
        The cpp node of the target constant value.
    speedup: ModelSpeedup
        The Model speedup module.
    Returns
    -------
    value: int/float/tensor
        The constant values parsed from the node.
    """
    logger.debug('Try to parse the constant value: %s', cvalue.debugName())
    if cvalue.toIValue() is not None:
        return cvalue.toIValue()
    if cvalue.debugName() in speedup.internal_result:
        return speedup.internal_result[cvalue.debugName()]
    # Get the operator node of the this value
    op_node = cvalue.node()

    inputs = op_node.inputs()
    input_values = [parse_constant(_i, speedup) for _i in inputs]
    func = trans_from_jit_to_python[op_node.kind()](op_node, speedup)
    return func(*input_values)


def dropout_python(node, speedup):
    return torch.dropout


def flatten_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    start_dim = inputs[1].toIValue()
    end_dim = inputs[2].toIValue()
    new_flatten = partial(torch.flatten, start_dim=start_dim, end_dim=end_dim)
    return new_flatten


def relu_inplace_python(node, speedup):
    return torch.relu_


def relu_python(node, speedup):
    return torch.relu


def sigmoid_python(node, speedup):
    return torch.sigmoid


def mean_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    keep_dim = inputs[2].toIValue()
    new_mean = partial(torch.mean, dim=tuple(dim_list), keepdim=keep_dim)
    return new_mean


def add_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    constant = None
    for i in range(2):
        input_i = inputs[i]
        debug_name = input_i.debugName()
        if debug_name not in speedup.internal_result:
            # this input is a constant value
            # TODO: what if this input is a constant tensor

            if input_i.toIValue() is not None:
                constant = parse_constant(input_i, speedup)
                break
    if constant is None:
        return torch.add
    else:
        new_add = partial(torch.add, constant)
        return new_add


def floor_div_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    divisor = inputs[1]
    constant = None
    if divisor.debugName() not in speedup.internal_result:
        # divisor is a constant value/tensor
        constant = parse_constant(divisor, speedup)
    if constant is None:
        return torch.floor_divide
    else:
        new_op = partial(torch.floor_divide, other=constant)
        return new_op


def mul_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    constant = None
    for i in range(2):
        input_i = inputs[i]
        debug_name = input_i.debugName()
        if debug_name not in speedup.internal_result:
            constant = parse_constant(input_i, speedup)
            # both two inputs cannot be constants at the same time
            break
    if constant is None:
        return torch.mul
    else:
        new_mul = partial(torch.mul, constant)
        return new_mul


def transpose_python(node, speedup):
    return torch.t


def transpose2_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_1 = inputs[1].toIValue()
    dim_2 = inputs[2].toIValue()
    new_transpose = partial(torch.transpose, dim0=dim_1, dim1=dim_2)
    return new_transpose


def matmul_python(node, speedup):
    return torch.matmul


def div_python(node, speedup):
    # The second input parameter of torch.div can be a
    # tensor or a constant, if it is a constant, we need
    # to return
    c_node = node.key_node
    inputs = list(c_node.inputs())
    if inputs[1].debugName() in speedup.internal_result:
        # the second input parameters is the output of the other
        # nodes
        return torch.div
    else:
        other = inputs[1].toIValue()
        new_div = partial(torch.div, other=other)

        return new_div


def softmax_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = inputs[1].toIValue()
    new_softmax = partial(torch.softmax, dim=dim)
    return new_softmax


def contiguous_python(node, speedup):
    class contiguousModule(torch.nn.Module):
        def forward(self, x):
            return x.contiguous()
    return contiguousModule()


def gelu_python(node, speedup):
    return torch.nn.GELU()


def avgpool2d_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    kernel_size = translate_list(inputs[1], speedup)
    stride = translate_list(inputs[2], speedup)
    padding = translate_list(inputs[3], speedup)
    new_avgpool = partial(torch.nn.functional.avg_pool2d,
                          kernel_size=kernel_size, stride=stride, padding=padding)
    return new_avgpool


def adaptive_avgpool_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    output_size = translate_list(inputs[1], speedup)
    new_avgpool = torch.nn.AdaptiveAvgPool2d(output_size)
    return new_avgpool


def tupleunpack_python(node, speedup):
    # Note: tuple unpack should only exists at the
    # the end of the model, and is no need to replace/propagate mask
    return None


def num2tensor_python(node, speedup):
    return torch.nn.Identity()


def exp_python(node, speedup):
    return torch.exp


def squeeze_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = None
    if len(inputs) > 1:
        dim = parse_constant(inputs[1], speedup)
    new_squeeze = partial(torch.squeeze, dim=dim)
    return new_squeeze

def unsqueeze_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = parse_constant(inputs[1], speedup)
    new_unsqueeze = partial(torch.unsqueeze, dim=dim)
    return new_unsqueeze

##########################################################
# Split Line
# Following module/functions cannot be translated into a
# single function, so we use torch.nn.Module to wrap the
# the core function, and return the torch.nn.Module instead
##########################################################


def slice_python(node, speedup):
    class SliceMoudle(torch.nn.Module):
        def __init__(self, sliceobj):
            super(SliceMoudle, self).__init__()
            self.sliceobj = sliceobj

        def forward(self, x, *args):
            # args is for the slice dimension and indexes, however,
            # we already get them from the cpp nodes. Note, though, we
            # don't need the slice indexes any more, we cannot remove this
            # parameter here, because, there may be multiple inputs passed from
            # previous nodes such as aten::size
            logger.info('Model has Slice operation, and the operand size=%s, Slice object:%s', str(
                x.size()), str(self.sliceobj))
            return x[self.sliceobj]

    c_node = node.key_node
    inputs = list(c_node.inputs())

    slice_dim = parse_constant(inputs[1], speedup)
    slice_start = parse_constant(inputs[2], speedup)
    slice_end = parse_constant(inputs[3], speedup)
    slice_step = parse_constant(inputs[4], speedup)
    slice_obj = slice(slice_start, slice_end, slice_step)
    slice_list = []
    for _ in range(slice_dim):
        slice_list.append(slice(None, None))
    logger.info('Slice dim:%s, Slice obj:%s', str(slice_dim), str(slice_obj))
    slice_list.append(slice_obj)
    return SliceMoudle(tuple(slice_list))


def select_python(node, speedup):
    class SelectModule(torch.nn.Module):
        def __init__(self, dim, index):
            super(SelectModule, self).__init__()
            self.dim = dim
            self.index = index

        def forward(self, x):
            return x.select(self.dim, self.index)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = inputs[1].toIValue()
    index = inputs[2].toIValue()
    return SelectModule(dim, index)


def size_python(node, speedup):
    # return None
    class SizeMoudle(torch.nn.Module):
        def __init__(self, sizedim):
            super(SizeMoudle, self).__init__()
            self.sizedim = sizedim

        def forward(self, x):
            return torch.as_tensor([x.size(self.sizedim)], dtype=torch.long)
            # return torch.tensor(x.size(self.sizedim))
    c_node = node.key_node
    inputs = list(c_node.inputs())
    size_dim = inputs[1].toIValue()
    return SizeMoudle(size_dim)


def toint_python(node, speedup):
    class ToIntModule(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.int)
    return ToIntModule()


def view_python(node, speedup):
    class ViewModule(torch.nn.Module):
        def __init__(self, shape):
            super(ViewModule, self).__init__()
            self.shape = shape
            logger.info('View Module output size: %s', str(self.shape))

        def forward(self, *args):
            return args[0].view(self.shape)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    shape = translate_list(inputs[1], speedup)
    return ViewModule(shape)


def reshape_python(node, speedup):
    class ReshapeModule(torch.nn.Module):
        def __init__(self, shape):
            super(ReshapeModule, self).__init__()
            self.shape = shape
            logger.info('Reshape Module output size: %s', str(self.shape))

        def forward(self, *args):
            return args[0].view(self.shape)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    shape = translate_list(inputs[1], speedup)
    return ReshapeModule(shape)


def permute_python(node, speedup):
    class PermuteModule(torch.nn.Module):
        def __init__(self, dimlist):
            super(PermuteModule, self).__init__()
            self.dimlist = dimlist

        def forward(self, x):
            return x.permute(self.dimlist)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    return PermuteModule(dim_list)


def getattr_python(node, speedup):
    """
    Note: Ops started with Prim:: is not taken as the key node,
    so we directly pass the Cpp node into this funciton.
    Parameters
    ----------
    node: torch._C.Node
        The cpp node of prim::Getattr
    speedup: ModelSpeedup
        The corresponding speedup object.
    """
    class GetModule(torch.nn.Module):
        def __init__(self, key):
            super(GetModule, self).__init__()
            self.key = key

        def forward(self, obj):
            logger.info('Get attribute: %s', self.key)
            return getattr(obj, self.key)
    # get the name of the attribute, for example
    # prim::GetAttr[name="module_list"](%self.1)
    assert node.kind() == 'prim::GetAttr'
    pattern = '\[name=\"(.*?)\"\]'
    key_words = re.findall(pattern, str(node))
    assert len(key_words) == 1
    return GetModule(key_words[0])


def upsample_bilinear2d_python(node, speedup):
    class UpsampleModule(torch.nn.Module):
        def __init__(self, size_list, scale_list):
            super(UpsampleModule, self).__init__()
            self.size_list = size_list
            self.scale_list = scale_list

        def forward(self, *args):
            """
            The first input of args is the target tensor to upsample
            , the following parameters is useless, because we already
            get the size_list and the scale_list by parsing the cpp_nodes.
            """
            return torch.nn.functional.upsample_bilinear(args[0],
                                                         size=self.size_list, scale_factor=self.scale_list)
    c_node = node.key_node
    inputs = list(c_node.inputs())
    size_list_node = inputs[1].node()
    scale_list_node = inputs[3].node()
    size_list = None
    scale_list = None

    if size_list_node.kind() == 'prim::ListConstruct':
        size_list = translate_list(inputs[1], speedup)
    if scale_list_node.kind() == 'prim::ListConstruct':
        scale_list = translate_list(inputs[3], speedup)
    return UpsampleModule(size_list, scale_list)


def typeas_python(node, speedup):
    """
    currently only support type_as float.
    TODO: support more types in the type_as, need to figure out
    how to get the scalar type from torch._C.TensorType.
    """
    class TypeasModule(torch.nn.Module):
        def __init__(self, dtype=torch.float):
            self.example = torch.zeros(1, dtype=dtype)

        def forward(self, x):
            return x.type_as(self.example)
    return TypeasModule()


def to_python(node, speedup):
    # for the time being, only device parameters are supported
    class ToModule(torch.nn.Module):
        def __init__(self, device):
            super(ToModule, self).__init__()

        def forward(self, x):
            return x.to(device)

    c_node = node.key_node
    inputs = list(c_node.inputs())
    device = inputs[3].toIValue()
    return ToModule(device)


def cat_python(node, speedup):
    class CatModule(torch.nn.Module):
        def __init__(self, cat_dim):
            super(CatModule, self).__init__()
            self.cat_dim = cat_dim

        def forward(self, *args):
            return torch.cat(args, dim=self.cat_dim)

    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim = inputs[1].toIValue()
    return CatModule(dim)


trans_from_jit_to_python = {
    'aten::add': add_python,
    'aten::add_': add_python,
    'aten::mul': mul_python,
    'aten::mul_': mul_python,
    'aten::relu': relu_python,
    'aten::relu_': relu_inplace_python,
    'aten::sigmoid': sigmoid_python,
    'aten::sigmoid_': sigmoid_python,
    # tanh behaives like relu
    'aten::tanh': relu_python,
    'aten::tanh_': relu_python,
    'aten::flatten': flatten_python,
    'aten::mean': mean_python,
    'aten::dropout': dropout_python,
    'aten::slice': slice_python,
    'aten::select': select_python,
    'aten::size': size_python,
    'aten::t': transpose_python,
    'aten::transpose': transpose2_python,
    'aten::Int': toint_python,
    'aten::view': view_python,
    'aten::reshape': reshape_python,
    'aten::permute': permute_python,
    'aten::matmul': matmul_python,
    'aten::div': div_python,
    'aten::floor_divide': floor_div_python,
    'aten::softmax': softmax_python,
    'aten::contiguous': contiguous_python,
    'aten::gelu': gelu_python,
    'aten::cat': cat_python,
    'aten::avg_pool2d': avgpool2d_python,
    'aten::max_pool2d': avgpool2d_python,
    'aten::adaptive_avg_pool2d': adaptive_avgpool_python,
    'aten::to': to_python,
    'aten::type_as': typeas_python,
    'aten::upsample_bilinear2d': upsample_bilinear2d_python,
    'aten::exp': exp_python,
    'aten::squeeze': squeeze_python,
    'aten::unsqueeze': unsqueeze_python,
    'prim::TupleUnpack': tupleunpack_python,
    'prim::ListUnpack': tupleunpack_python,
    'prim::NumToTensor': num2tensor_python,
    'prim::GetAttr': getattr_python

}


def jit_to_python_function(node, speedup):
    """
    Return a callable object to inference the mask according to the
    node.op_type.

    Parameters
    ---------
    node: NodeGroup
        The target node to inference the mask
    speedup: ModelSpeedup
        The speedup object of the target model.

    Returns
    ------
    func: callable object(nn.Module/function)
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in trans_from_jit_to_python:
        logger.error(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', node.op_type)
        # return None to skip the mask inference for this node
        return None
    return trans_from_jit_to_python[node.op_type](node, speedup)
