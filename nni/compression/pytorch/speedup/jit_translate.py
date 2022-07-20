# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Dict, List, Type
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only imports the below statements during type checking
    from nni.compression.pytorch.speedup import ModelSpeedup
    from nni.common.graph_utils import NodePyGroup

import re
import logging
from functools import partial
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# to exclude partial

__all__ = [
    'getattr_python', 'jit_to_python_function', 'num2tensor_python', 'parse_constant', 'slice_python',
    'translate_list', 'tupleunpack_python', 'dtype_trans', 'memory_format_trans'
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
    global trans_func_dict
    func = trans_func_dict[op_node.kind()](op_node, speedup)
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
    undetermined: List[int | str]
        A list of the right positions of arguments.
        Position is an int in positional or a str in keyword.
    special_treat: Dict[int | str, Callable]
        A Dict of the positions and methods.
        The values of these positions should be treat by those methods.

    """
    def __new__(cls, func: Callable, positional: List[Any], keyword: Dict[str, Any],
                undetermined: List[int | str], special_treat: Dict[int | str, Callable]):
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
            logger.warning('throw some args away when calling the function \'%s\'', self.func.__name__)
        for i in range(0, len(self.undetermined)):
            p = self.undetermined[i]
            v = args[i]
            if type(p) is int:
                self.positional[p] = v
            else:
                self.keyword[p] = v

        for p, fs in self.special_treat.items():
            if type(p) is int:
                for f in fs: self.positional[p] = f(self.positional[p])
            else:
                for f in fs: self.keyword[p] = f(self.keyword[p])
        result = self.func(*self.positional, **self.keyword)
        if type(result) is int: # turn result of 'size' into tensor
            result = torch.as_tensor([result], dtype=torch.long)
        return result

# There are some types that will be convert into enums after jit.
# So we should recover them back:
#   device, dtype, layout, memory_format, qscheme, qengine, dispatchkey

enum2dtype_dict = {
    0: torch.uint8, # byte
    1: torch.int8, # char
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64, # double
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
    }
    enum2dtype_dict = {**enum2dtype_dict, **scalar2dtype_dict_qint}
if torch.__version__ < '1.11.0' or torch.__version__ >= '1.12.0':
    # torch.complex32 is disabled in 1.11
    enum2dtype_dict[8] = torch.complex32
if torch.__version__ >= '1.12.0':
    # In 1.12, torch.quint2x4 is existed, and there is a 'QUInt2x4Storage'.
    enum2dtype_dict[17] = torch.quint2x4
def dtype_trans(ivalue: int | torch.dtype):
    """
    Special process for dtype.
    Torch will transform dtype to an enum in cpp, so the value of dtype we get in jit is an int.
    This function is used to recover the int to torch.dtype in python.

    Parameters
    ----------
    ivalue:
        The value of dtype or method to be recovered.

    """
    if ivalue is None or type(ivalue) is torch.dtype:
        return ivalue
    elif type(ivalue) is int:
        global enum2dtype_dict
        if ivalue not in enum2dtype_dict:
            raise TypeError("Unimplemented scalar type")
        return enum2dtype_dict[ivalue]
    else:
        raise TypeError("Unimplemented scalar type")

enum2memory_format_dict = {
    0: torch.contiguous_format,
    1: torch.preserve_format, # char
    2: torch.channels_last,
    3: torch.channels_last_3d,
}
def memory_format_trans(ivalue: int | torch.memory_format):
    """
    Special process for memory_format.
    Torch will transform memory_format to an enum in cpp, so the value of memory_format we get in jit is an int.
    This function is used to recover the int to torch.memory_format in python.

    Parameters
    ----------
    ivalue:
        The value of memory_format or method to be recovered.

    """
    if ivalue is None or type(ivalue) is torch.memory_format:
        return ivalue
    elif type(ivalue) is int:
        global enum2memory_format_dict
        if ivalue not in enum2memory_format_dict:
            raise TypeError("Unimplemented memory_format type")
        return enum2memory_format_dict[ivalue]
    else:
        raise TypeError("Unimplemented memory_format type")

special_treat_dict = {
    'dtype': dtype_trans,
    'memory_format': memory_format_trans,
}
def generate_aten_to_python(func: Callable, node: NodePyGroup, speedup: ModelSpeedup):
    """
    parse a
    Return a callable object to inference the mask according to the
    node.op_type.

    Parameters
    ---------
    func: Callable
        The torch function one-to-one correspondence with the node.
    node: NodePyGroup
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
    special_treat = dict() # for dtype trans now

    i = schema.find('(')
    if schema[i + 1] != ',':
        i += 1
        j = schema.find(',', i)
        while j != -1:
            if schema[j - 1] != '*':
                # detect and it to the special_treat
                match = re.search("(\w+)=[^\s]+$", schema[i:j])
                if not match:
                    match = re.search("(\w+)$", schema[i:j])
                arg_name = match.group(1)
                if arg_name in special_treat_dict:
                    key = positional_num
                    if key not in special_treat: special_treat[key] = [special_treat_dict[arg_name]]
                    else: special_treat[key].append(special_treat_dict[arg_name])
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
                    arg_name = match.group(1)
                    keyword_list.append(arg_name)
                    # detect and it to the special_treat
                    if arg_name in special_treat_dict:
                        key = arg_name
                        if key not in special_treat: special_treat[key] = [special_treat_dict[arg_name]]
                        else: special_treat[key].append(special_treat_dict[arg_name])

                    i = j + 1
                    j = schema.find(',', i)
                break

    ## translate inputs, to positional, keyword and undetermined
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
    undetermined_special_treat = dict()
    for p, fs in special_treat.items():
        if p in undetermined:
            undetermined_special_treat[p] = fs
        elif type(p) is int:
            for f in fs: positional[p] = f(positional[p])
        else:
            for f in fs: keyword[p] = f(keyword[p])

    return my_partial(func, positional, keyword, undetermined, undetermined_special_treat)

trans_func_dict = {
    'aten::slice': slice_python,
    'aten::Int': partial(generate_aten_to_python, torch._C._TensorBase.int),
    'prim::TupleUnpack': tupleunpack_python,
    'prim::ListUnpack': tupleunpack_python,
    'prim::NumToTensor': num2tensor_python,
    'prim::GetAttr': getattr_python,
}
def init_add_functions(func_from: ModuleType | Type):
    """
    Add function/method attributes from a module/class, to the trans_func_dict

    Parameters
    ---------
    func_from: module/class
        The module/class include needed functions

    """
    global trans_func_dict
    new_trans_func_dict = dict()
    for name in dir(func_from):
        attr = getattr(func_from, name)
        if callable(attr) and not name.startswith("__"):
            new_trans_func_dict['aten::' + name] = partial(generate_aten_to_python, attr)
    trans_func_dict = {**new_trans_func_dict, **trans_func_dict}

init_add_functions(torch._C._VariableFunctions)
init_add_functions(torch._C._nn)
init_add_functions(torch._C._TensorBase)

def jit_to_python_function(node: NodePyGroup, speedup: ModelSpeedup):
    """
    Return a callable object to inference the mask according to the
    node.op_type.

    Parameters
    ---------
    node: NodePyGroup
        The target node to inference the mask
    speedup: ModelSpeedup
        The speedup object of the target model.

    Returns
    ------
    func: callable object(nn.Module/function)
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    global trans_func_dict
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in trans_func_dict:
        logger.error(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', node.op_type)
        # return None to skip the mask inference for this node
        return None
    return trans_func_dict[node.op_type](node, speedup)
