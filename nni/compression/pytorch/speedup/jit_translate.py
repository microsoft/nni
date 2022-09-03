# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Type, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only imports the below statements during type checking
    from nni.compression.pytorch.speedup import ModelSpeedup
    from nni.common.graph_utils import NodePyGroup

import re
import logging
from functools import partial, lru_cache
import copy
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
jitid_2_dtype = {4: torch.long, 6:torch.float32}

# to exclude partial

__all__ = [
    'getattr_python', 'jit_to_python_function', 'num2tensor_python', 'parse_constant', 'slice_python',
    'translate_list', 'tupleunpack_python', 'dtype_trans', 'memory_format_trans'
]

def translate_list(list_node: torch._C.Value, speedup: ModelSpeedup=None) -> List:
    """
    Get the list of values from the list construct node.

    Parameters
    ----------
    list_node
        The cpp node of the target list.
    speedup
        The Module speedup module.

    Returns
    -------
    values
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
            values.append(speedup.internal_result[debugName])
        else:
            # if the corresponding value is a constant
            values.append(_i.toIValue())
    return values

def parse_constant(cvalue: torch._C.Value, speedup: ModelSpeedup) -> Any:
    """
    Parse the constant values from this Node

    Parameters
    ----------
    cvalue
        The cpp node of the target constant value.
    speedup
        The Model speedup module.

    Returns
    -------
    value
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
    if op_node.kind() not in trans_func_dict:
        raise RuntimeError('Unsupported function op node type: {}'.format(op_node.kind()))

    func = trans_func_dict[op_node.kind()](op_node, speedup)
    return func(*input_values)

def slice_python(node: NodePyGroup, speedup: ModelSpeedup):
    class SliceMoudle(torch.nn.Module):
        def __init__(self, sliceobj):
            super(SliceMoudle, self).__init__()
            # we need to deepcopy the value here, because, in the
            # follwing steps, we may randomize the input tensor
            # which will change the values of the sliceobj
            self.sliceobj = copy.deepcopy(sliceobj)

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

    if inputs[0].debugName() not in speedup.internal_result:
        # The inputs of slice operator may be the constant
        target_tensor = parse_constant(inputs[0], speedup)
        slice_list = tuple(slice_list)

        def constant_slice(*args):
            return target_tensor[slice_list]
        return constant_slice
    else:
        return SliceMoudle(tuple(slice_list))

def cat_python(node: NodePyGroup, speedup: ModelSpeedup):
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

def tupleunpack_python(_node: NodePyGroup, _speedup: ModelSpeedup) -> Optional[Callable]:
    # Note: tuple unpack should only exists at the
    # the end of the model, and is no need to replace/propagate mask
    return None

def num2tensor_python(_node: NodePyGroup, _speedup: ModelSpeedup):
    return torch.nn.Identity()

def getattr_python(node: NodePyGroup, _speedup: ModelSpeedup):
    """
    Note: Ops started with Prim:: is not taken as the key node,
    so we directly pass the Cpp node into this funciton.

    Parameters
    ----------
    node
        The cpp node of prim::Getattr
    speedup
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

class FuncAdapter:
    """
    A function adapter which can reorder arguments.
    It can be initialate with constant argument, and positions of each non-constant
    argument. When called, it can put arguments into correct position, then call the
    function.

    Attributes
    ----------
    func
        The function or method to be called.
    positional
        Positional arguments values. The placeholder is None if it's non-constant.
    keyword
        Keyword arguments values. The placeholder is None if it's non-constant.
    undetermined
        A list of the right positions of arguments.
        Position is an int in positional or a str in keyword.
    special_treat
        A Dict of the positions and methods.
        The values of these positions should be treat by those methods.

    """

    def __init__(self, func: Callable, positional: List[Any], keyword: Dict[str, Any],
                undetermined: List[Union[int, str]], special_treat: Dict[Union[int, str], Callable]):
        if not callable(func):
            raise TypeError('the "func" argument must be callable')

        self.func = func
        self.positional = positional
        self.keyword = keyword
        self.undetermined = undetermined
        self.special_treat = special_treat

    def __call__(self, *args):
        assert len(args) >= len(self.undetermined)
        if len(args) > len(self.undetermined):
            logger.warning('throw some args away when calling the function "%s"', self.func.__name__)

        for i, p in enumerate(self.undetermined):
            v = args[i]
            if isinstance(p, int):
                self.positional[p] = v
            else:
                self.keyword[p] = v

        for p, fs in self.special_treat.items():
            if isinstance(p, int):
                for f in fs:
                    self.positional[p] = f(self.positional[p])
            else:
                for f in fs:
                    self.keyword[p] = f(self.keyword[p])
        result = self.func(*self.positional, **self.keyword)
        if isinstance(result, int): # turn result of 'size' into tensor
            result = torch.as_tensor([result], dtype=torch.long)
        return result

# There are some types that will be convert into enums after jit.
# So we should recover them back:
#   device, dtype, layout, memory_format, qscheme, qengine, dispatchkey

enum_to_dtype_names = {
    0: 'uint8',
    1: 'int8',
    2: 'int16',
    3: 'int32',
    4: 'int64',
    5: 'float16',
    6: 'float32',
    7: 'float64',
    8: 'complex32',
    9: 'complex64',
    10: 'complex128',
    11: 'bool',
    12: 'qint8',
    13: 'quint8',
    14: 'qint32',
    15: 'bfloat16',
    16: 'quint4x2',
    17: 'quint2x4',
}

enum_to_dtype_dict = {}

for enum_value, dtype_name in enum_to_dtype_names.items():
    if hasattr(torch, dtype_name):
        enum_to_dtype_dict[enum_value] = getattr(torch, dtype_name)

def dtype_trans(ivalue: Union[int, torch.dtype]):
    """
    Special process for dtype.
    Torch will transform dtype to an enum in cpp, so the value of dtype we get in jit is an int.
    This function is used to recover the int to torch.dtype in python.

    Parameters
    ----------
    ivalue
        The value of dtype or method to be recovered.

    """
    if ivalue is None or isinstance(ivalue, torch.dtype):
        return ivalue
    elif isinstance(ivalue, int):
        if ivalue in enum_to_dtype_dict:
            return enum_to_dtype_dict[ivalue]
    raise TypeError('No torch.dtype corresponding to the value "%s"', ivalue)

enum_to_memory_format_dict = {
    0: torch.contiguous_format,
    1: torch.preserve_format,
    2: torch.channels_last,
    3: torch.channels_last_3d,
}

def memory_format_trans(ivalue: Union[int, torch.memory_format]):
    """
    Special process for memory_format.
    Torch will transform memory_format to an enum in cpp, so the value of memory_format we get in jit is an int.
    This function is used to recover the int to torch.memory_format in python.

    Parameters
    ----------
    ivalue
        The value of memory_format or method to be recovered.

    """
    if ivalue is None or isinstance(ivalue, torch.memory_format):
        return ivalue
    elif isinstance(ivalue, int):
        global enum_to_memory_format_dict
        if ivalue in enum_to_memory_format_dict:
            return enum_to_memory_format_dict[ivalue]
    raise TypeError('No torch.memory_format corresponding to the value "%s"', ivalue)

special_treat_dict = {
    'dtype': dtype_trans,
    'memory_format': memory_format_trans,
}

schema_fix_dict = {
    # functinon 'to', 'randint', and 'sparse_coo_tensor' has different schema between python and c++.
    # https://pytorch.org/docs/stable/jit_unsupported.html#ops-with-divergent-schemas-between-torch-python
    """aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Ten
    sor(a))""":
        """aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, *, int? memory_format=None)
         -> (Tensor(a))""",
    'aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))':
        'aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, *, int? memory_format=None) -> (Tensor(a))',
    'aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))':
        'aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, *, int? memory_format=None) -> (Tensor(a))',

    # todo: are the arguments 'pin_memory' and 'requires_grad' related?
    # functions in the python have only 'requires_grad' and functions in the aten have only 'pin_memory'

    # 'aten::randint(int high, int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)',
    # """aten::randint.generator(int high, int[] size, *, Generator? generator, int? dtype=None, int? layout=None, Device? device=None, boo
    # l? pin_memory=None) -> (Tensor)""",
    # """aten::randint.low(int low, int high, int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None)
    # -> (Tensor)""",
    # """aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, int? dtype=None, int? layout=None, Device? dev
    # ice=None, bool? pin_memory=None) -> (Tensor)""",

    # """aten::sparse_coo_tensor.size(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=False) -> (Te
    # nsor)""",
    # """aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, int? dtype=None, int? layout=None, Device? device=None, bool? pi
    # n_memory=None) -> (Tensor)""",
    # """aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, int? dtype=None, int? layout=None, Device? devi
    # ce=None, bool? pin_memory=None) -> (Tensor"""'
}
@lru_cache(maxsize=256)
def parse_aten_schema(schema: str):
    """
    Parse the schema, to positional_num and keyword_list, and detect if the argument should be specially treated.
    """
    if schema in schema_fix_dict:
        schema = schema_fix_dict[schema]

    positional_num = 0
    keyword_list = list()
    special_treat = dict() # for dtype and memory_format trans now

    for arg in torch._C.parse_schema(schema).arguments:
        if not arg.kwarg_only:
            key = positional_num
            positional_num += 1
        else:
            key = arg.name
            keyword_list.append(key)

        if arg.name in special_treat_dict:
            if key not in special_treat:
                special_treat[key] = [special_treat_dict[arg.name]]
            else:
                special_treat[key].append(special_treat_dict[arg.name])

    return positional_num, keyword_list, special_treat

def parse_input_value(speedup: ModelSpeedup, input_nodes: List[torch._C.Node], positional_num: int, keyword_list: List[str]):
    """
    translate inputs, to constant positional arguments, constant keyword arguments, and undetermined positions
    """
    positional = list()
    keyword = dict()
    undetermined = list()

    for ainput in input_nodes:
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
    return positional, keyword, undetermined

def special_treat_to_constant_value(positional: List, keyword: Dict[str], undetermined: List[Union[int, str]],
                                    special_treat: Dict[Union[int, str], Callable]):
    """
    if any argument with special_treat is not in undetermined, do the treat
    """
    undetermined_special_treat = dict()
    for p, fs in special_treat.items():
        if p in undetermined:
            undetermined_special_treat[p] = fs
        elif isinstance(p, int):
            for f in fs: positional[p] = f(positional[p])
        else:
            for f in fs: keyword[p] = f(keyword[p])
    return undetermined_special_treat

def generate_aten_to_python(func: Callable, node: NodePyGroup, speedup: ModelSpeedup) -> FuncAdapter:
    """
    parse a Return a callable object to inference the mask according to the node.op_type.

    Parameters
    ---------
    func
        The torch function one-to-one correspondence with the node.
    node
        The target node to inference the mask
    speedup
        The speedup object of the target model.

    Returns
    ------
    func
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    c_node = node.key_node

    schema = c_node.schema()
    positional_num, keyword_list, special_treat = parse_aten_schema(schema)

    input_nodes = list(c_node.inputs())
    positional, keyword, undetermined = parse_input_value(speedup, input_nodes, positional_num, keyword_list)

    undetermined_special_treat = special_treat_to_constant_value(positional, keyword, undetermined, special_treat)

    return FuncAdapter(func, positional, keyword, undetermined, undetermined_special_treat)

trans_func_dict = {
    'aten::slice': slice_python,
    'aten::cat': cat_python,
    'aten::Int': partial(generate_aten_to_python, torch._C._TensorBase.int),

    'prim::TupleUnpack': tupleunpack_python,
    'prim::ListUnpack': tupleunpack_python,
    'prim::NumToTensor': num2tensor_python,
    'prim::GetAttr': getattr_python,
}
def init_add_functions(func_from: Union[ModuleType, Type[Any]]):
    """
    Add function/method attributes from a module/class, to the trans_func_dict

    Parameters
    ---------
    func_from
        The module/class include needed functions

    """
    global trans_func_dict
    new_trans_func_dict = dict()
    for name in dir(func_from):
        attr = getattr(func_from, name)
        if callable(attr) and not name.startswith('__'):
            new_trans_func_dict['aten::' + name] = partial(generate_aten_to_python, attr)
    trans_func_dict = {**new_trans_func_dict, **trans_func_dict}

init_add_functions(torch._C._VariableFunctions)
init_add_functions(torch._C._nn)
init_add_functions(torch._C._TensorBase)

def jit_to_python_function(node: NodePyGroup, speedup: ModelSpeedup) -> FuncAdapter:
    """
    Return a callable object to inference the mask according to the node.op_type.

    Parameters
    ---------
    node
        The target node to inference the mask
    speedup
        The speedup object of the target model.

    Returns
    ------
    func
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    logger.debug(
        'Translate C function %s into its python version', node.op_type)
    if node.op_type not in trans_func_dict:
        logger.error(
            '%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', node.op_type)
        # return None to skip the mask inference for this node
        return None
    return trans_func_dict[node.op_type](node, speedup)
