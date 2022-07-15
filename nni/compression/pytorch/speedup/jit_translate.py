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
    'getattr_python', 'jit_to_python_function', 'num2tensor_python', 'parse_constant', 'slice_python', 'torch',
    'translate_list', 'tupleunpack_python',
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
    func = get_trans_dict()[op_node.kind()](op_node, speedup)
    return func(*input_values)


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
    """
    Yet another functools.partial which support customed positions of arguements
    Array with associated photographic information.

    Attributes
    ----------
    func: Callable
        The function or method to be called.
    positional: List
        Positional arguments.
    keyword: dict
        Keyword arguments.
    undetermined: List
        A list of the right positions of arguments.
        Position is an int in positional or a str in keyword.
    special_treat: List
        A list of the positions and methods.
        The values of these positions should be treat by those methods.

    """
    def __new__(cls, func, positional, keyword, undetermined, special_treat):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            positional = func.positional + positional
            keyword = {**func.keyword, **keyword}
            func = func.func

        self = super(my_partial, cls).__new__(cls)

        self.func = func
        self.positional = positional
        self.keyword = keyword
        self.undetermined = undetermined
        self.special_treat = special_treat
        return self

    def __call__(self, /, *args):
        assert len(args) >= len(self.undetermined)
        if len(args) > len(self.undetermined):
            logger.warning('drop some args when calling the function')
        for i in range(0, len(self.undetermined)):
            p = self.undetermined[i]
            v = args[i]
            if type(p) is int:
                self.positional[p] = v
            else:
                self.keyword[p] = v

        for (p, f) in self.special_treat:
            if type(p) is int:
                self.positional[p] = f(self.positional[p])
            else:
                self.keyword[p] = f(self.keyword[p])
        result = self.func(*self.positional, **self.keyword)
        if type(result) is int: # turn result of 'size' into tensor
            result = torch.as_tensor([result], dtype=torch.long)
        return result
            

scalar2dtype_dict = {
    0: torch.uint8, # byte
    1: torch.int8, # char
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64, # double
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    15: torch.bfloat16,
}
if torch.__version__ >= '1.9.0':
    scalar2dtype_dict_qint = {
        12: torch.qint8,
        13: torch.quint8,
        14: torch.qint32,
        16: torch.quint4x2,
        17: torch.quint2x4,
    }
    scalar2dtype_dict = {**scalar2dtype_dict, **scalar2dtype_dict_qint}
def dtype_trans(scalar_type):
    """
    Special process for dtype.
    Torch will transform dtype to an enum in cpp, so the value of dtype we get in jit is an int.
    This function is used to recover the int to torch.dtype in python.

    Parameters
    ----------
    scalar_type: 
        The value of dtype or method to be recovered.

    """
    if scalar_type is None or type(scalar_type) is torch.dtype:
        return scalar_type
    elif type(scalar_type) is int:
        global scalar2dtype_dict
        if scalar_type not in scalar2dtype_dict:
            raise TypeError("Unimplemented scalar type")
        return scalar2dtype_dict[scalar_type]
    else:
        raise TypeError("Unimplemented scalar type")

def generate_aten_to_python(func, node, speedup):
    """
    parse a
    Return a callable object to inference the mask according to the
    node.op_type.

    Parameters
    ---------
    func: Callable
        The torch function one-to-one correspondence with the node.
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
    c_node = node.key_node
    schema = c_node.schema()

    ## parse the schema, to positional_num and keyword_list
    schema = schema[0:schema.rfind(') ->')] + ','
    positional_num = 0
    keyword_list = list()
    special_treat = list() # for dtype trans now

    i = schema.find('(')
    if schema[i + 1] != ',':
        i += 1
        j = schema.find(',', i)
        while j != -1:
            if schema[j - 1] != '*':
                if 'dtype=None' in schema[i:j]:
                    special_treat.append((positional_num, dtype_trans))
                positional_num += 1

                i = j + 1
                j = schema.find(',', i)
            else:
                i = j + 1
                j = schema.find(',', i)
                while j != -1:
                    match = re.search("(\w+)=[^\s]+$", schema[i:j])
                    if not match:
                        match = re.search("(\w+)$", schema[i:j])
                    key = match.group(1)
                    keyword_list.append(key)
                    if 'dtype=None' in schema[i:j]:
                        assert key == 'dtype'
                        special_treat.append((key, dtype_trans))

                    i = j + 1
                    j = schema.find(',', i)
                break

    ## trans the input, to positional, keyword and undetermined
    positional = list()
    keyword = dict()
    undetermined = list()

    for ainput in list(c_node.inputs()):
        if ainput.node().kind() == 'prim::ListConstruct':
            arg = translate_list(ainput, speedup)
        elif ainput.node().kind() == 'prim::Constant':
            arg = ainput.toIValue()
        else:
            assert 'aten::' in ainput.node().kind() or 'prim::' in ainput.node().kind()
            if len(positional) < positional_num:
                undetermined.append(len(positional))
            else:
                undetermined.append(keyword_list[positional_num - len(positional)])
            arg = None

        if len(positional) < positional_num:
            positional.append(arg)
        else:
            keyword[keyword_list[positional_num - len(positional)]] = arg

    ## if something in special_treat is not in undetermined, do the treat
    undetermined_special_treat = list()
    for (p, f) in special_treat:
        if p in undetermined:
            undetermined_special_treat.append((p, f))
        elif type(p) is int:
            positional[p] = f(positional[p])
        else:
            keyword[p] = f(keyword[p])

    return my_partial(func, positional, keyword, undetermined, special_treat)

members = None
def get_trans_dict():
    """
    Get the string to aten_recover_func dict.

    """
    global members
    if not members:
        members = {
            'aten::slice': slice_python,
            'aten::Int': partial(generate_aten_to_python, torch._C._TensorBase.int),
            'prim::TupleUnpack': tupleunpack_python,
            'prim::ListUnpack': tupleunpack_python,
            'prim::NumToTensor': num2tensor_python,
            'prim::GetAttr': getattr_python,
        }
        def init_add_functions(func_from):
            global members
            new_members = dict()
            for name in dir(func_from):
                attr = getattr(func_from, name)
                if callable(attr) and not name.startswith("__"):
                    new_members['aten::' + name] = partial(generate_aten_to_python, attr)
            members = {**new_members, **members}

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
    trans_dict = get_trans_dict()
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in trans_dict:
        logger.error(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', node.op_type)
        # return None to skip the mask inference for this node
        return None
    return trans_dict[node.op_type](node, speedup)
