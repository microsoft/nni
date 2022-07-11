# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import logging
from functools import partial
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# to exclude partial

__all__ = [
    'adaptive_avgpool_python', 'add_python', 'avgpool2d_python', 'cat_python', 'contiguous_python',
    'div_python', 'dropout_python', 'exp_python', 'flatten_python', 'floor_div_python', 'gelu_python',
    'getattr_python', 'jit_to_python_function', 'matmul_python', 'mean_python',
    'mul_python', 'num2tensor_python', 'parse_constant', 'permute_python', 'relu_inplace_python',
    'relu_python', 'reshape_python', 'select_python', 'sigmoid_python', 'size_python', 'slice_python',
    'softmax_python', 'squeeze_python', 'to_python', 'toint_python', 'torch', 'trans_from_jit_to_python',
    'translate_list', 'transpose2_python', 'transpose_python', 'tupleunpack_python', 'typeas_python',
    'unsqueeze_python', 'upsample_bilinear2d_python', 'view_python', 'sum_python'
]


def translate_list(list_node, speedup=None):
    """
    Get the list of values from the list construct node.

    Parameters
    ----------
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


def sum_python(node, speedup):
    c_node = node.key_node
    inputs = list(c_node.inputs())
    dim_list = translate_list(inputs[1], speedup)
    keep_dim = inputs[2].toIValue()
    new_sum = partial(torch.sum, dim=tuple(dim_list), keepdim=keep_dim)
    return new_sum

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

def toint_python(node, speedup):
    class ToIntModule(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.int)
    return ToIntModule()

def tupleunpack_python(node, speedup):
    # Note: tuple unpack should only exists at the
    # the end of the model, and is no need to replace/propagate mask
    return None

def num2tensor_python(node, speedup):
    return torch.nn.Identity()

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

class my_partial:
    def __new__(cls, func, undetermined, args, keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(my_partial, cls).__new__(cls)

        self.func = func
        self.undetermined = undetermined
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args):
        assert len(args) == len(self.undetermined)
        for i in range(0, len(args)):
            p = self.undetermined[i]
            v = args[i]
            if type(p) is int:
                self.args[p] = v
            else:
                self.keywords[p] = v
        return self.func(*self.args, **self.keywords)

def generate_aten_to_python(func, node, speedup):
    c_node = node.key_node
    schema = c_node.schema()

    #, , , *, xx=xx, xx) ==> num_before_star, keyword_list
    #, , ) ==> num_before_star, keywords = {}
    schema = schema[0:schema.rfind(') ->')] + ','
    num_before_star = 0
    keyword_list = list()
    i = schema.find(',')
    while i != -1:
        if schema[i - 1] != '*':
            num_before_star += 1
            i = schema.find(',', i + 1)
        else:
            i += 1
            j = schema.find(',', i)
            while j != -1:
                match = re.search("(\w+)=[^\s]+$", schema[i:j])
                if not match:
                    match = re.search("(\w+)$", schema[i:j])
                keyword_list.append(match.group(1))
                i = j + 1
                j = schema.find(',', i)
            break
    args = list()
    keywords = dict()
    undetermined = list()
    
    for input in list(c_node.inputs()):
        if input.node().kind() == 'prim::ListConstruct':
            arg = translate_list(input, speedup)
        elif input.node().kind() == 'prim::Constant':
            arg = input.toIValue()
        else:
            assert 'aten::' in input.node().kind() or 'prim::' in input.node().kind()
            if len(args) < num_before_star:
                undetermined.append(len(args))
            else:
                undetermined.append(keyword_list[num_before_star - len(args)])
            arg = None

        if len(args) < num_before_star:
            args.append(arg)
        else:
            keywords[keyword_list[num_before_star - len(args)]] = arg
    new_func = my_partial(func, undetermined, args, keywords)
    return new_func

members = None
def init_trans_dict():
    global members
    if not members:
        members = {
            'aten::slice': slice_python, # cannot find function or method 'slice' under torch._C
            'prim::TupleUnpack': tupleunpack_python,
            'prim::ListUnpack': tupleunpack_python,
            'prim::NumToTensor': num2tensor_python,
            'prim::GetAttr': getattr_python,
        }
        def init_add_functions(func_from):
            global members
            new_members = {"aten::" + attr : partial(generate_aten_to_python, getattr(func_from, attr)) for attr in dir(func_from) if attr not in members and callable(getattr(func_from, attr)) and not attr.startswith("__")}
            members = {**members, **new_members}
        
        init_add_functions(torch._C._VariableFunctions)
        init_add_functions(torch._C._nn)
        init_add_functions(torch._C._TensorBase)
        
    return members
    

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
    the_dict = init_trans_dict()
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in the_dict:
        logger.error(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', node.op_type)
        # return None to skip the mask inference for this node
        return None
    return the_dict[node.op_type](node, speedup)
