from typing import (Any, Dict, List)

from . import debug_configs

__all__ = ['Operation', 'Cell']

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


def _convert_name(name: str) -> str:
    """
    Convert the names using separator '.' to valid variable name in code
    """
    return name.replace('.', '__')


class Operation:
    """
    Calculation logic of a graph node.

    The constructor is private. Use `Operation.new()` to create operation object.

    `Operation` is a naive record.
    Do not "mutate" its attributes or store information relate to specific node.
    All complex logic should be implemented in `Node` class.

    Attributes
    ----------
    type
        Operation type name (e.g. Conv2D).
        If it starts with underscore, the "operation" is a special one (e.g. subgraph, input/output).
    parameters
        Arbitrary key-value parameters (e.g. kernel_size).
    """

    def __init__(self, type_name: str, parameters: Dict[str, Any], _internal: bool = False):
        assert _internal, '`Operation()` is private, use `Operation.new()` instead'
        self.type: str = type_name
        self.parameters: Dict[str, Any] = parameters

    def to_init_code(self, field: str) -> str:
        raise NotImplementedError()

    def to_forward_code(self, field: str, output: str, inputs: List[str]) -> str:
        raise NotImplementedError()

    def _to_class_name(self) -> str:
        raise NotImplementedError()

    def __bool__(self) -> bool:
        return True

    @staticmethod
    def new(type_name: str, parameters: Dict[str, Any] = {}, cell_name: str = None) -> 'Operation':
        if type_name == '_cell':
            # NOTE: cell_name is the same as its Node's name, when the cell is wrapped within the node
            return Cell(cell_name, parameters)
        else:
            if debug_configs.framework.lower() in ('torch', 'pytorch'):
                from .operation_def import torch_op_def  # pylint: disable=unused-import
                cls = PyTorchOperation._find_subclass(type_name)
            elif debug_configs.framework.lower() in ('tf', 'tensorflow'):
                from .operation_def import tf_op_def  # pylint: disable=unused-import
                cls = TensorFlowOperation._find_subclass(type_name)
            else:
                raise ValueError(f'Unsupported framework: {debug_configs.framework}')
            return cls(type_name, parameters, _internal=True)

    @classmethod
    def _find_subclass(cls, subclass_name):
        for subclass in cls.__subclasses__():
            if subclass.__name__ == subclass_name:
                return subclass
        return cls

    def __repr__(self):
        type_name = type(self).__name__
        args = [f'{key}={repr(value)}' for key, value in self.parameters.items()]
        if type_name != self.type:
            args = [f'type="{self.type}"'] + args
        return f'{type_name}({", ".join(args)})'

    def __eq__(self, other):
        return type(other) is type(self) and other.type == self.type and other.parameters == self.parameters


class PyTorchOperation(Operation):
    def _to_class_name(self) -> str:
        if self.type.startswith('__torch__.'):
            return self.type[len('__torch__.'):]
        elif self.type.startswith('__mutated__.'):
            return self.type[len('__mutated__.'):]
        else:
            return None

    def get_import_pkg(self) -> str:
        if self.type.startswith('__torch__.'):
            return self.type[len('__torch__.'):].split('.')[0]
        elif self.type.startswith('__mutated__.'):
            return self.type[len('__mutated__.'):].split('.')[0]
        else:
            return None

    def to_init_code(self, field: str) -> str:
        if self._to_class_name() is not None:
            assert 'positional_args' not in self.parameters
            kw_params = ', '.join(f'{key}={repr(value)}' for key, value in self.parameters.items())
            return f'self.{field} = {self._to_class_name()}({kw_params})'
        return None

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any] = None) -> str:
        """
        Parameters
        ----------
        field : str
            the name of member submodule
        output : str
            the output name (lvalue) of this line of code
        inputs : List[str]
            variables used in this line of code
        inputs_value : List[Any]
            some variables are actually constant, their real values are recorded in ```inputs_value```.
            if not constant, we simply put None at the corresponding index
        """
        from .converter.op_types import OpTypeName
        if self._to_class_name() is not None:
            return f'{output} = self.{field}({", ".join(inputs)})'
        elif self.type == 'shared':
            return f'{output} = self.{field}({", ".join(inputs)})'
        elif self.type.startswith('Function.'):
            func_name = self.type[len('Function.'):]
            return f'{output} = F.{func_name}({", ".join(inputs)})'
        elif self.type == 'prim::Constant':
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
            print('zql value: ', value, type(value))
        elif self.type == 'prim::ListConstruct':
            return f'{output} = [{", ".join(inputs)}]'
        elif self.type == 'prim::TupleConstruct':
            return f'{output} = ({", ".join(inputs)})'
        elif self.type == 'prim::TupleUnpack':
            # have single output here, because the following code uses index to access the unpacked values
            assert len(inputs) == 1
            return f'{output} = {inputs[0]}'
        elif self.type == 'prim::GetAttr':
            if self.parameters['value'] is not None:
                return f"{output} = {self.parameters['value']}"
            else:
                return f"{output} = {self.parameters['input']}.{self.parameters['name']}"
        elif self.type == 'prim::is_cuda':
            return f'{output} = {inputs[0]}.is_cuda'
        elif self.type == 'aten::mean':
            return f'{output} = torch.mean({inputs[0]}, {", ".join(inputs[1:-1])}, out={inputs[-1]})'
        elif self.type == 'aten::__getitem__':
            assert len(inputs) == 2
            return f'{output} = {inputs[0]}[{inputs[1]}]'
        elif self.type == 'aten::append':
            assert len(inputs) == 2
            return f'_, {output} = {inputs[0]}.append({inputs[1]}), {inputs[0]}'
        elif self.type == 'aten::cat':
            assert len(inputs) == 2
            return f'{output} = torch.cat({inputs[0]}, dim={inputs[1]})'
        elif self.type == 'aten::add':
            # TODO: verify the correctness
            #return f'{output} = ' + ' + '.join(inputs)
            if len(inputs) == 2:
                return f'{output} = {inputs[0]}.add({inputs[1]})'
            else:
                assert len(inputs) == 3
                return f'{output} = {inputs[0]}.add({inputs[1]}, alpha={inputs[2]})'
        elif self.type == 'aten::add_':
            if len(inputs) == 2:
                return f'{output} = {inputs[0]}.add_({inputs[1]})'
            else:
                assert len(inputs) == 3
                return f'{output} = {inputs[0]}.add_({inputs[1]}, alpha={inputs[2]})'
        elif self.type == OpTypeName.MergedSlice:
            assert (len(inputs) - 1) % 4 == 0
            slices = []
            dim = int((len(inputs) - 1) / 4)
            for i in range(dim):
                slices.append(f'{inputs[i*4+2]}:{inputs[i*4+3]}:{inputs[i*4+4]}')
            slice_str = ','.join(slices)
            return f'{output} = {inputs[0]}[{slice_str}]'
        elif self.type == 'aten::size':
            if len(inputs) == 2:
                return f'{output} = {inputs[0]}.size({inputs[1]})'
            else:
                return f'{output} = {inputs[0]}.size()'
        elif self.type == 'aten::view':
            assert len(inputs) == 2
            return f'{output} = {inputs[0]}.view({inputs[1]})'
        elif self.type == 'aten::reshape':
            assert len(inputs) == 2
            return f'{output} = {inputs[0]}.reshape({inputs[1]})'
        elif self.type == 'aten::slice':
            raise RuntimeError('not supposed to have aten::slice operation')
        elif self.type == 'aten::Bool':
            return f'{output} = bool({inputs[0]})'
        elif self.type == 'aten::flatten':
            return f'{output} = torch.flatten({inputs[0]}, {inputs[1]}, {inputs[2]})'
        elif self.type == 'aten::sigmoid':
            assert len(inputs) == 1
            return f'{output} = torch.sigmoid({inputs[0]})'
        elif self.type == 'aten::__not__':
            return f'{output} = not {inputs[0]}'
        elif self.type == 'aten::transpose':
            return f'{output} = {inputs[0]}.transpose({inputs[1]}, {inputs[2]})'
        elif self.type == 'aten::contiguous':
            # defined in pytorch/c10/core/MemoryFormat.h
            assert inputs_value[1] in [0, 1, 2]
            return f'{output} = {inputs[0]}.contiguous(memory_format={mem_format[inputs_value[1]]})'
        elif self.type == 'aten::detach':
            return f'{output} = {inputs[0]}.detach()'
        elif self.type == 'aten::new_full':
            device_str = f', device=torch.device({inputs[5]})' if inputs_value[5] is not None else ''
            dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[3]]}' if inputs_value[3] is not None else ''
            return f'{output} = {inputs[0]}.new_full({inputs[1]}, {inputs[2]}{dtype_str}{device_str})'
        elif self.type == 'aten::new_empty':
            device_str = f', device=torch.device({inputs[4]})' if inputs_value[4] is not None else ''
            dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[2]]}' if inputs_value[2] is not None else ''
            return f'{output} = {inputs[0]}.new_empty({inputs[1]}{dtype_str}{device_str})'
        elif self.type == 'aten::new_zeros':
            # in pytorch: new_zeros(size, dtype=None, device=None, requires_grad=False) → Tensor
            # in aten: - func: new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None,
            # Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
            # TODO: check requires_grad when it is true!!!
            device_str = f', device=torch.device({inputs[4]})' if inputs_value[4] is not None else ''
            dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[2]]}' if inputs_value[2] is not None else ''
            return f'{output} = {inputs[0]}.new_zeros({inputs[1]}{dtype_str}{device_str})'
        elif self.type == 'aten::tensor':
            device_str = f', device=torch.device({inputs[2]})' if inputs_value[2] is not None else ''
            dtype_str = f', dtype={scalar_type_to_pytorch_type[inputs_value[1]]}' if inputs_value[1] is not None else ''
            req_grad_str = f', requires_grad={inputs[3]}' if inputs_value[3] else ''
            return f'{output} = torch.tensor({inputs[0]}{dtype_str}{device_str}{req_grad_str})'
        elif self.type == 'aten::abs':
            return f'{output} = {inputs[0]}.abs()'
        elif self.type == 'aten::abs_':
            return f'{output} = {inputs[0]}.abs_()'
        elif self.type == 'aten::acos':
            return f'{output} = {inputs[0]}.acos()'
        elif self.type == 'aten::acos_':
            return f'{output} = {inputs[0]}.acos_()'
        elif self.type == 'aten::asin':
            return f'{output} = {inputs[0]}.asin()'
        elif self.type == 'aten::atan':
            return f'{output} = {inputs[0]}.atan()'
        elif self.type == 'aten::atan2':
            return f'{output} = {inputs[0]}.atan2({inputs[1]})'
        elif self.type == 'aten::addbmm':
            return f'{output} = {inputs[0]}.addbmm({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'
        elif self.type == 'aten::baddbmm':
            return f'{output} = {inputs[0]}.baddbmm({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'
        elif self.type == 'aten::addcdiv':
            return f'{output} = {inputs[0]}.addcdiv({inputs[1]}, {inputs[2]}, value={inputs[3]})'
        elif self.type == 'aten::addcmul':
            return f'{output} = {inputs[0]}.addcmul({inputs[1]}, {inputs[2]}, value={inputs[3]})'
        elif self.type == 'aten::addmm':
            return f'{output} = {inputs[0]}.addmm({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'
        elif self.type == 'aten::addmv':
            return f'{output} = {inputs[0]}.addmv({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'
        elif self.type == 'aten::bmm':
            return f'{output} = {inputs[0]}.bmm({inputs[1]})'
        elif self.type == 'aten::addr':
            return f'{output} = {inputs[0]}.addr({inputs[1]}, {inputs[2]}, beta={inputs[3]}, alpha={inputs[4]})'
        elif self.type == 'aten::allclose':
            return f'{output} = {inputs[0]}.allclose({inputs[1]}, rtol={inputs[2]}, atol={inputs[3]}, equal_nan={inputs[4]})'
        elif self.type == 'aten::angle':
            return f'{output} = {inputs[0]}.angle()'
        elif self.type == 'aten::argmax':
            return f'{output} = {inputs[0]}.argmax(dim={inputs[1]}, keepdim={inputs[2]})'
        elif self.type == 'aten::argmin':
            return f'{output} = {inputs[0]}.argmin(dim={inputs[1]}, keepdim={inputs[2]})'
        elif self.type == 'aten::argsort':
            return f'{output} = {inputs[0]}.argsort(dim={inputs[1]}, descending={inputs[2]})'
        elif self.type == 'aten::bernoulli':
            assert inputs_value[1] is None
            return f'{output} = {inputs[0]}.bernoulli()'
        elif self.type == 'aten::bincount':
            return f'{output} = {inputs[0]}.bincount(weights={inputs[1]}, minlength={inputs[2]})'
        elif self.type == 'aten::bitwise_not':
            return f'{output} = {inputs[0]}.bitwise_not()'
        elif self.type == 'aten::bitwise_and':
            return f'{output} = {inputs[0]}.bitwise_and({inputs[1]})'
        elif self.type == 'aten::bitwise_or':
            return f'{output} = {inputs[0]}.bitwise_or({inputs[1]})'
        elif self.type == 'aten::bitwise_xor':
            return f'{output} = {inputs[0]}.bitwise_xor({inputs[1]})'
        elif self.type == 'noop_identity':
            # this operator type is added by us
            return f'{output} = {", ".join(inputs)}'
        else:
            raise RuntimeError(f'unsupported operation type: {self.type} ? {self._to_class_name()}')


class TensorFlowOperation(Operation):
    def _to_class_name(self) -> str:
        return 'K.layers.' + self.type


class Cell(PyTorchOperation):
    """
    TODO: this is pytorch cell

    An operation reference to a subgraph.

    Example code:
    ```
        def __init__(...):
            ...
            self.cell = CustomCell(...)
            self.relu = K.layers.ReLU()
            ...

        def forward(...):
            ...
            x = self.cell(x)
            ...
    ```

    In above example, node `self.cell`'s operation is `Cell(cell_name='CustomCell')`.
    For comparison, `self.relu`'s operation is `Operation(type='ReLU')`.

    TODO: parameters of subgraph (see `Node` class)

    Attributes
    ----------
    type
        Always "_cell".
    parameters
        A dict with only one item; the key is "cell" and the value is cell's name.
    framework
        No real usage. Exists for compatibility with base class.
    """

    def __init__(self, cell_name: str, parameters: Dict[str, Any] = {}):
        self.type = '_cell'
        self.cell_name = cell_name
        self.parameters = parameters

    def _to_class_name(self):
        # TODO: ugly, think about how to refactor this part
        return _convert_name(self.cell_name)


class _IOPseudoOperation(Operation):
    """
    This is the pseudo operation used by I/O nodes.
    The benefit is that users no longer need to verify `Node.operation is not None`,
    especially in static type checking.
    """

    def __init__(self, type_name: str, io_names: List = None):
        assert type_name.startswith('_')
        super(_IOPseudoOperation, self).__init__(type_name, {}, True)
        self.io_names = io_names

    def to_init_code(self, field: str) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def to_forward_code(self, field: str, output: str, inputs: List[str]) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def __bool__(self) -> bool:
        return False
