# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import (Any, List)

import torch

from ..operation import PyTorchOperation


mem_format = [
    'torch.contiguous_format',      # 0
    'torch.preserve_format',        # 1
    'torch.channels_last',          # 2
]

# this snippet is copied from torch/onnx/symbolic_helper.py,
# the original definition is in c10/core/ScalarType.h
# This indicates each scalar type's corresponding
scalar_type_to_pytorch_type = [
    'torch.uint8',        # 0
    'torch.int8',         # 1
    'torch.short',        # 2
    'torch.int',          # 3
    'torch.int64',        # 4
    'torch.half',         # 5
    'torch.float',        # 6
    'torch.double',       # 7
    'torch.complex32',    # 8
    'torch.complex64',    # 9
    'torch.complex128',   # 10
    'torch.bool',         # 11
]

class NoOpIdentity(PyTorchOperation):
    """
    this operator type is added by us
    """
    _ori_type_name = ['noop_identity']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {", ".join(inputs)}'

class ModuleOperator(PyTorchOperation):
    _ori_type_name = ['ModuleOperator', 'shared']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = self.{field}({", ".join(inputs)})'

class FunctionalOperator(PyTorchOperation):
    _ori_type_name = ['FunctionalOperator']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        func_name = self.type[len('Function.'):]
        if not hasattr(torch.nn.functional, func_name):
            raise RuntimeError('For now, we only support calling independent functions from `torch.nn.functional`, '
                               f'{func_name} is not in it.')
        return f'{output} = F.{func_name}({", ".join(inputs)})'

class PrimConstant(PyTorchOperation):
    _ori_type_name = ['prim::Constant']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # TODO: refactor this part, maybe we can remove the code gen of prim::Constant
        # TODO: deal with all the types
        if self.parameters['type'] == 'None':
            return f'{output} = None'
        elif self.parameters['type'] in ('int', 'float', 'bool', 'int[]'):
            return f'{output} = {self.parameters["value"]}'
        elif self.parameters['type'] == 'str':
            str_val = self.parameters["value"]
            return f'{output} = "{str_val}"'
        elif self.parameters['type'] == 'Device':
            value = self.parameters['value']
            return f'{output} = torch.device("{value}")'
        elif self.parameters['type'] in ('dict', 'list', 'tuple'):
            # TODO: prim::TupleIndex is not supported yet
            return f'{output} = {repr(self.parameters["value"])}'
        else:
            raise RuntimeError(f'unsupported type of prim::Constant: {self.parameters["type"]}')

class PrimListConstruct(PyTorchOperation):
    _ori_type_name = ['prim::ListConstruct']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = [{", ".join(inputs)}]'

class PrimListUnpack(PyTorchOperation):
    _ori_type_name = ['prim::ListUnpack']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}'

class PrimTupleConstruct(PyTorchOperation):
    _ori_type_name = ['prim::TupleConstruct']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = ({", ".join(inputs)})'

class PrimTupleUnpack(PyTorchOperation):
    _ori_type_name = ['prim::TupleUnpack']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # have single output here, because the following code uses index to access the unpacked values
        assert len(inputs) == 1
        return f'{output} = {inputs[0]}'

class PrimGetAttr(PyTorchOperation):
    _ori_type_name = ['prim::GetAttr']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        if self.parameters['value'] is not None:
            return f"{output} = {self.parameters['value']}"
        else:
            return f"{output} = {self.parameters['input']}.{self.parameters['name']}"

class SimpleMember(PyTorchOperation):
    _ori_type_name = ['prim::is_cuda', 'prim::data']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        member_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{member_name}'

class AtenContiguous(PyTorchOperation):
    _ori_type_name = ['aten::contiguous']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # defined in pytorch/c10/core/MemoryFormat.h
        assert inputs_value[1] in [0, 1, 2]
        return f'{output} = {inputs[0]}.contiguous(memory_format={mem_format[inputs_value[1]]})'

class AtenGetitem(PyTorchOperation):
    _ori_type_name = ['aten::__getitem__']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'{output} = {inputs[0]}[{inputs[1]}]'

class AtenAppend(PyTorchOperation):
    _ori_type_name = ['aten::append']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'_, {output} = {inputs[0]}.append({inputs[1]}), {inputs[0]}'

class MergedSlice(PyTorchOperation):
    _ori_type_name = ['MergedSlice']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        if (len(inputs) - 1) % 4 == 0:
            slices = []
            dim = int((len(inputs) - 1) / 4)
            for i in range(dim):
                slices.append(f'{inputs[i*4+2]}:{inputs[i*4+3]}:{inputs[i*4+4]}')
            slice_str = ','.join(slices)
            return f'{output} = {inputs[0]}[{slice_str}]'
        elif len(inputs) == 4:
            # this case is for simple list
            return f'{output} = {inputs[0]}[{inputs[1]}:{inputs[2]}:{inputs[3]}]'
        else:
            raise RuntimeError('Unsupported slice pattern')

# the following Aten classes means these aten ops are not in torch.Tensor

class AtenBool(PyTorchOperation):
    _ori_type_name = ['aten::Bool']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = bool({inputs[0]})'

class AtenNot(PyTorchOperation):
    _ori_type_name = ['aten::__not__']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = not {inputs[0]}'

class AtenCat(PyTorchOperation):
    _ori_type_name = ['aten::cat']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'{output} = torch.cat({inputs[0]}, dim={inputs[1]})'

#====================================

class AtenTensors(PyTorchOperation):
    _ori_type_name = ['aten::full', 'aten::full_like', 'aten::empty_like',
                      'aten::ones_like', 'aten::zeros_like', 'aten::rand',
                      'aten::randn', 'aten::scalar_tensor', 'aten::new_full',
                      'aten::new_empty', 'aten::new_zeros', 'aten::arange',
                      'aten::tensor', 'aten::ones', 'aten::zeros']

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        schemas = torch._C._jit_get_schemas_for_operator(self.type)
        # match number of inputs
        overloaded_defs = [len(s.arguments) for s in schemas]
        matched = overloaded_defs.index(len(inputs))
        args_list = []
        for idx, arg in enumerate(schemas[matched].arguments):
            if arg.name == 'dtype':
                arg_str = f'dtype={scalar_type_to_pytorch_type[inputs_value[idx]]}' if inputs_value[idx] is not None else ''
            elif arg.name == 'layout':
                if inputs_value[idx] is not None:
                    arg_str = f'layout=torch.strided'
                    print('Warning: only support `torch.strided` for now!!!')
                else:
                    arg_str = ''
            elif arg.name == 'device':
                arg_str = f'device=torch.device({inputs[idx]})' if inputs_value[idx] is not None else ''
            elif arg.name == 'memory_format':
                arg_str = f'memory_format={mem_format[inputs_value[idx]]}' if inputs_value[idx] is not None else ''
            elif arg.name == 'pin_memory':
                # TODO: deal with this argument
                continue
            elif arg.name == 'requires_grad':
                arg_str = f'requires_grad={inputs[idx]}' if inputs_value[idx] else ''
            elif str(arg.type).startswith('Optional['):
                arg_str = f'{arg.name}={inputs[idx]}'
            else:
                arg_str = f'{inputs[idx]}'
            if arg_str != '':
                args_list.append(arg_str)
        op_name = self.type.split('::')[-1]
        if hasattr(torch, op_name):
            return f'{output} = torch.{op_name}({", ".join(args_list)})'
        else:
            return f'{output} = {inputs[0]}.{op_name}({", ".join(args_list[1:])})'

#====================================

class AtenFloordiv(PyTorchOperation):
    _ori_type_name = ['aten::floordiv']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]} // {inputs[1]}'

class AtenLen(PyTorchOperation):
    _ori_type_name = ['aten::len']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = len({inputs[0]})'

class AtenIntImplicit(PyTorchOperation):
    _ori_type_name = ['aten::IntImplicit', 'aten::Float', 'aten::Int', 'aten::ScalarImplicit']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        if self.type.endswith('Implicit'):
            return f'{output} = {inputs[0]}'
        elif self.type == 'aten::Int':
            return f'{output} = int({inputs[0]})'
        elif self.type == 'aten::Float':
            return f'{output} = float({inputs[0]})'

class AtenIndex(PyTorchOperation):
    _ori_type_name = ['aten::index']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}[{inputs[1]}]'

ManuallyChooseDef = {
    'aten::flatten': [('start_dim', 'int', '0'), ('end_dim', 'int', '-1')],
    'aten::split': [('split_size', 'int', 'None'), ('dim', 'int', '0')]
}

TensorOpExceptions = {
    'aten::sub': lambda output, inputs: f'{output} = {inputs[0]} - {inputs[1]}', # example: x.size(1) - 3
    'aten::add': lambda output, inputs: f'{output} = {inputs[0]} + {inputs[1]}' # example: input.shape[0] + 5
}

TorchOpExclude = ['aten::Size', 'aten::as_tensor', 'aten::device',
                  'aten::manual_seed', 'aten::quantized_gru', 'aten::quantized_lstm',
                  'aten::save', 'aten::tensor', 'aten::wait'
]

def _hidden(name):
    return name.startswith('_') and not name.startswith('__')

def _emit_args(args):
    # filter out the `out` argument here
    return [(arg.name, str(arg.type), str(arg.default_value)) for arg in args] #  if arg.name != 'out'

def _get_tensor_ops():
    def is_tensor_method(schema):
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != 'self':
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True

    op_args = {}
    # discover methods
    for elem in dir(torch.Tensor):
        if not _hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    op_name = 'aten::' + elem
                    args = _emit_args(schema.arguments[1:])
                    if op_name in op_args:
                        op_args[op_name].append(args)
                    else:
                        op_args[op_name] = [args]

    return op_args.keys(), op_args

def _get_torch_ops():
    torch_op_args = {}
    for mod in torch.jit._builtins._modules_containing_builtins:
        name = mod.__name__
        if name == 'torch._C._nn':
            continue
        # only process 'torch.XXX'
        for elem in dir(mod):
            builtin = torch.jit._builtins._find_builtin(getattr(mod, elem))
            if builtin is not None:
                schemas = torch._C._jit_get_schemas_for_operator(builtin)
                for schema in schemas:
                    # remove _tan but not __and__
                    if not _hidden(elem):
                        op_name = 'aten::' + elem
                        if len(schema.arguments) > 0 and schema.arguments[0].name == 'self':
                            continue
                        args = _emit_args(schema.arguments)
                        if op_name in torch_op_args:
                            torch_op_args[op_name].append(args)
                        else:
                            torch_op_args[op_name] = [args]

    return torch_op_args.keys(), torch_op_args

def _get_torch_ops_exclude_tensor_ops():
    tensor_op_names, _ = _get_tensor_ops()
    torch_op_names, torch_ops = _get_torch_ops()

    torch_exclude_ops = {}
    for name in torch_op_names:
        if name not in tensor_op_names:
            if name not in TorchOpExclude:
                # exclude the ops that are not in
                # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
                torch_exclude_ops[name] = torch_ops[name]

    return torch_exclude_ops.keys(), torch_exclude_ops

class TensorOps(PyTorchOperation):
    """
    corresponding to _get_tensor_ops in torch.jit.supported_ops
    """
    _ori_type_name, _op_args = _get_tensor_ops()

    comparison_ops = {'aten::eq': '==', 'aten::ne': '!=', 'aten::le': '<=', 'aten::ge': '>=', 'aten::lt': '<', 'aten::gt': '>'}

    @staticmethod
    def _get_matched_args(_type, inputs):
        def has_same_arg_name(matched):
            concated_names = []
            for i, each in enumerate(matched):
                name = ','.join([arg[0] for arg in each])
                concated_names.append(name)
            for i in range(len(concated_names) - 1):
                if concated_names[i] != concated_names[i+1]:
                    return False
            return True

        overloaded_defs = TensorOps._op_args[_type]
        matched = []
        for each in overloaded_defs:
            # plus 1 because we skip the first argument when generating tensor op def
            if len(each) + 1 == len(inputs):
                matched.append(each)
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            # TODO: match with arg's type. manually choose for now
            if has_same_arg_name(matched):
                # return any one is okay
                return matched[0]
            elif _type in ManuallyChooseDef:
                return ManuallyChooseDef[_type]
            else:
                raise RuntimeError(f'tensor op type {_type} has more than one matched: {matched}')
        else:
            if _type in TensorOpExceptions:
                return None
            raise RuntimeError(f'tensor op type {_type} has no matched')

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # TODO: deal with conditional ops
        if self.type in TensorOps.comparison_ops:
            return f'{output} = ({inputs[0]} {TensorOps.comparison_ops[self.type]} {inputs[1]})'
        matched_args = TensorOps._get_matched_args(self.type, inputs)
        if matched_args is None:
            return TensorOpExceptions[self.type](output, inputs)
        op_name = self.type.split('::')[-1]
        args_str = ', '.join([f'{name}={inputs[i+1]}' for i, (name, t, default) in enumerate(matched_args)])
        return f'{output} = {inputs[0]}.{op_name}({args_str})'

class TorchOps(PyTorchOperation):
    """
    corresponding to _get_nn_functional_ops in torch.jit.supported_ops
    """
    _ori_type_name, _op_args = _get_torch_ops_exclude_tensor_ops()
    # add 'aten::pixel_shuffle'
    _op_args['aten::pixel_shuffle'] = [[('input', 'Tensor', 'None'), ('upscale_factor', 'Optional[int]', 'None')]]
    _ori_type_name = _op_args.keys()

    @staticmethod
    def _get_matched_args(_type, inputs):
        def has_same_arg_name(matched):
            concated_names = []
            for i, each in enumerate(matched):
                name = ','.join([arg[0] for arg in each])
                concated_names.append(name)
            for i in range(len(concated_names) - 1):
                if concated_names[i] != concated_names[i+1]:
                    return False
            return True

        overloaded_defs = TorchOps._op_args[_type]
        matched = []
        for each in overloaded_defs:
            if len(each) == len(inputs):
                matched.append(each)
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            # TODO: match with arg's type. manually choose for now
            if has_same_arg_name(matched):
                # return any one is okay
                return matched[0]
            else:
                raise RuntimeError(f'torch op type {_type} has more than one matched: {matched}')
        else:
            raise RuntimeError(f'torch op type {_type} has no matched')

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        matched_args = TorchOps._get_matched_args(self.type, inputs)
        op_name = self.type.split('::')[-1]
        args_str = ', '.join([f'{name}={inputs[i]}' if t.startswith('Optional[') else f'{inputs[i]}' \
            for i, (name, t, default) in enumerate(matched_args)])
        return f'{output} = torch.{op_name}({args_str})'

class AtenAvgpool2d(PyTorchOperation):
    # NOTE: it is not included in the above aten ops for unkown reason
    _ori_type_name = ['aten::avg_pool2d']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = F.avg_pool2d({", ".join(inputs)})'