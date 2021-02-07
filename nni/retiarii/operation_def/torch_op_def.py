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

class NoopIdentity(PyTorchOperation):
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
        return f'{output} = F.{func_name}({", ".join(inputs)})'

class PrimConstant(PyTorchOperation):
    _ori_type_name = ['prim::Constant']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # TODO: refactor this part, maybe we can remove the code gen of prim::Constant
        # TODO: deal with all the types
        if self.parameters['type'] == 'None':
            return f'{output} = None'
        elif self.parameters['type'] in ('int', 'float', 'bool'):
            return f'{output} = {self.parameters["value"]}'
        elif self.parameters['type'] == 'str':
            str_val = self.parameters["value"]
            return f'{output} = "{str_val}"'
        elif self.parameters['type'] == 'Device':
            value = self.parameters['value']
            return f'{output} = torch.device("{value}")'
        else:
            raise RuntimeError(f'unsupported type of prim::Constant: {self.parameters["type"]}')

class PrimListConstruct(PyTorchOperation):
    _ori_type_name = ['prim::ListConstruct']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = [{", ".join(inputs)}]'

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
    _ori_type_name = ['prim::is_cuda']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        member_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{member_name}'

class AtenContiguous(PyTorchOperation):
    _ori_type_name = ['aten::contiguous']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # defined in pytorch/c10/core/MemoryFormat.h
        assert inputs_value[1] in [0, 1, 2]
        return f'{output} = {inputs[0]}.contiguous(memory_format={mem_format[inputs_value[1]]})'

class AtenNewFull(PyTorchOperation):
    _ori_type_name = ['aten::new_full']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        device_str = f', device=torch.device({inputs[5]})' if inputs_value[5] is not None else ''
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[3]]}' if inputs_value[3] is not None else ''
        return f'{output} = {inputs[0]}.new_full({inputs[1]}, {inputs[2]}{dtype_str}{device_str})'

class AtenNewEmpty(PyTorchOperation):
    _ori_type_name = ['aten::new_empty']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        device_str = f', device=torch.device({inputs[4]})' if inputs_value[4] is not None else ''
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[2]]}' if inputs_value[2] is not None else ''
        return f'{output} = {inputs[0]}.new_empty({inputs[1]}{dtype_str}{device_str})'

class AtenNewZeros(PyTorchOperation):
    _ori_type_name = ['aten::new_zeros']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # in pytorch: new_zeros(size, dtype=None, device=None, requires_grad=False) → Tensor
        # in aten: - func: new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None,
        # Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        # TODO: check requires_grad when it is true!!!
        device_str = f', device=torch.device({inputs[4]})' if inputs_value[4] is not None else ''
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[2]]}' if inputs_value[2] is not None else ''
        return f'{output} = {inputs[0]}.new_zeros({inputs[1]}{dtype_str}{device_str})'

class AtenTensor(PyTorchOperation):
    _ori_type_name = ['aten::tensor']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        device_str = f', device=torch.device({inputs[2]})' if inputs_value[2] is not None else ''
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[1]]}' if inputs_value[1] is not None else ''
        req_grad_str = f', requires_grad={inputs[3]}' if inputs_value[3] else ''
        return f'{output} = torch.tensor({inputs[0]}{dtype_str}{device_str}{req_grad_str})'

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
        assert (len(inputs) - 1) % 4 == 0
        slices = []
        dim = int((len(inputs) - 1) / 4)
        for i in range(dim):
            slices.append(f'{inputs[i*4+2]}:{inputs[i*4+3]}:{inputs[i*4+4]}')
        slice_str = ','.join(slices)
        return f'{output} = {inputs[0]}[{slice_str}]'

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

class AtenFull(PyTorchOperation):
    _ori_type_name = ['aten::full', 'aten::full_like']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        device_str = f', device=torch.device({inputs[4]})' if inputs_value[4] is not None else ''
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[2]]}' if inputs_value[2] is not None else ''
        if inputs_value[3] is not None:
            layout_str = f', layout=torch.strided'
            print('Warning: only support `torch.strided` for now!!!')
        else:
            layout_str = ''
        if self.type == 'aten::full_like':
            mem_format_str = f', memory_format={mem_format[inputs_value[6]]}' if inputs_value[6] is not None else ''
            return f'{output} = torch.full_like({inputs[0]}, {inputs[1]}{dtype_str}{layout_str}{device_str}{mem_format_str})'
        else:
            return f'{output} = torch.full({inputs[0]}, {inputs[1]}{dtype_str}{layout_str}{device_str})'

class AtenEmptyLike(PyTorchOperation):
    _ori_type_name = ['aten::empty_like', 'aten::ones_like', 'aten::zeros_like']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[1]]}' if inputs_value[1] is not None else ''
        if inputs_value[2] is not None:
            layout_str = f', layout=torch.strided'
            print('Warning: only support `torch.strided` for now!!!')
        else:
            layout_str = ''
        device_str = f', device=torch.device({inputs[3]})' if inputs_value[3] is not None else ''
        mem_format_str = f', memory_format={mem_format[inputs_value[5]]}' if inputs_value[5] is not None else ''
        return f'{output} = torch.{op_name}({inputs[0]}{dtype_str}{layout_str}{device_str}{mem_format_str})'

class AtenRand(PyTorchOperation):
    _ori_type_name = ['aten::rand', 'aten::randn']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[1]]}' if inputs_value[1] is not None else ''
        if inputs_value[2] is not None:
            layout_str = f', layout=torch.strided'
            print('Warning: only support `torch.strided` for now!!!')
        else:
            layout_str = ''
        device_str = f', device=torch.device({inputs[3]})' if inputs_value[3] is not None else ''
        return f'{output} = torch.{op_name}({inputs[0]}{dtype_str}{layout_str}{device_str})'

class AtenFloordiv(PyTorchOperation):
    _ori_type_name = ['aten::floordiv']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]} // {inputs[1]}'

class AtenLen(PyTorchOperation):
    _ori_type_name = ['aten::len']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = len({inputs[0]})'

ManuallyChooseDef = {
    'aten::flatten': [('start_dim', 'int', '0'), ('end_dim', 'int', '-1')],
    'aten::split': [('split_size', 'int', 'None'), ('dim', 'int', '0')]
}

TensorOpExceptions = {
    'aten::sub': lambda output, inputs: f'{output} = {inputs[0]} - {inputs[1]}' # example: x.size(1) - 3
}

def _get_tensor_ops():
    def hidden(name):
        return name.startswith('_') and not name.startswith('__')

    def is_tensor_method(schema):
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != 'self':
            return False
        if not self.type.isSubtypeOf(torch._C.TensorType.get()):
            return False
        return True

    def emit_args(args):
        # filter out the `out` argument here
        return [(arg.name, str(arg.type), str(arg.default_value)) for arg in args] #  if arg.name != 'out'

    op_names = []
    op_args = {}
    # discover methods
    for elem in dir(torch.Tensor):
        if not hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    op_name = 'aten::' + elem
                    args = emit_args(schema.arguments[1:])
                    if op_name in op_args:
                        op_args[op_name].append(args)
                    else:
                        op_args[op_name] = [args]

    return op_args.keys(), op_args

class TensorOps(PyTorchOperation):
    """
    corresponding to _get_tensor_ops in torch.jit.supported_ops
    """
    _ori_type_name, _op_args = _get_tensor_ops()
    #print(_op_args)
    #exit(1)

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
                raise RuntimeError(f'type {_type} has more than one matched: {matched}')
        else:
            if _type in TensorOpExceptions:
                return None
            raise RuntimeError(f'type {_type} has no matched')

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        #print(self.type)
        #print(inputs)
        #print(TensorOps._op_args[self.type])
        #if self.type not in ValidationExceptions:
        #    assert len(inputs) == len(TensorOps._op_args[self.type]) + 1
        matched_args = TensorOps._get_matched_args(self.type, inputs)
        if matched_args is None:
            return TensorOpExceptions[self.type](output, inputs)
        op_name = self.type.split('::')[-1]
        args_str = ', '.join([f'{name}={inputs[i+1]}' for i, (name, t, default) in enumerate(matched_args)])
        print(args_str)
        return f'{output} = {inputs[0]}.{op_name}({args_str})'
