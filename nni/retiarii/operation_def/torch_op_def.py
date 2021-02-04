from typing import (Any, List)
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

class SimpleAten(PyTorchOperation):
    """
    It is for simple aten operators, these operators use simple rule to generate code.
    i.e., ```{inputs[0]}.op_name(", ".join(inputs[1:]))```
    """
    _ori_type_name = ['aten::ceil', 'aten::size', 'aten::view', 'aten::reshape',
        'aten::sigmoid', 'aten::transpose', 'aten::detach', 'aten::abs',
        'aten::acos', 'aten::asin', 'aten::atan', 'aten::atan2', 'aten::bmm',
        'aten::angle', 'aten::bitwise_not', 'aten::bitwise_and',
        'aten::bitwise_or', 'aten::bitwise_xor']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{op_name}({", ".join(inputs[1:])})'

class AtenAdd(PyTorchOperation):
    _ori_type_name = ['aten::addbmm', 'aten::baddbmm', 'aten::addmm', 'aten::addmv', 'aten::addr']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{op_name}({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'

class AtenAddc(PyTorchOperation):
    _ori_type_name = ['aten::addcdiv', 'aten::addcmul']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{op_name}({inputs[1]}, {inputs[2]}, value={inputs[3]})'

class AtenBernoulli(PyTorchOperation):
    _ori_type_name = ['aten::bernoulli']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert inputs_value[1] is None
        return f'{output} = {inputs[0]}.bernoulli()'

class AtenMean(PyTorchOperation):
    _ori_type_name = ['aten::mean']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = torch.mean({inputs[0]}, {", ".join(inputs[1:-1])}, out={inputs[-1]})'

class AtenSimpleAdd(PyTorchOperation):
    _ori_type_name = ['aten::add', 'aten::add_']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        # TODO: verify the correctness
        #return f'{output} = ' + ' + '.join(inputs)
        op_name = self.type.split('::')[-1]
        if len(inputs) == 2:
            return f'{output} = {inputs[0]}.{op_name}({inputs[1]})'
        else:
            assert len(inputs) == 3
            return f'{output} = {inputs[0]}.{op_name}({inputs[1]}, alpha={inputs[2]})'

class AtenArgmaxmin(PyTorchOperation):
    _ori_type_name = ['aten::argmax', 'aten::argmin']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        op_name = self.type.split('::')[-1]
        return f'{output} = {inputs[0]}.{op_name}(dim={inputs[1]}, keepdim={inputs[2]})'

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

class AtenBool(PyTorchOperation):
    _ori_type_name = ['aten::Bool']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = bool({inputs[0]})'

class AtenFlatten(PyTorchOperation):
    _ori_type_name = ['aten::flatten']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = torch.flatten({inputs[0]}, {inputs[1]}, {inputs[2]})'

class AtenNot(PyTorchOperation):
    _ori_type_name = ['aten::__not__']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = not {inputs[0]}'

class AtenAllclose(PyTorchOperation):
    _ori_type_name = ['aten::allclose']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}.allclose({inputs[1]}, rtol={inputs[2]}, atol={inputs[3]}, equal_nan={inputs[4]})'

class AtenArgsort(PyTorchOperation):
    _ori_type_name = ['aten::argsort']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}.argsort(dim={inputs[1]}, descending={inputs[2]})'

class AtenBincount(PyTorchOperation):
    _ori_type_name = ['aten::bincount']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {inputs[0]}.bincount(weights={inputs[1]}, minlength={inputs[2]})'

class NoopIdentity(PyTorchOperation):
    """
    this operator type is added by us
    """
    _ori_type_name = ['noop_identity']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        return f'{output} = {", ".join(inputs)}'

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

class AtenCat(PyTorchOperation):
    _ori_type_name = ['aten::cat']
    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        assert len(inputs) == 2
        return f'{output} = torch.cat({inputs[0]}, dim={inputs[1]})'

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
