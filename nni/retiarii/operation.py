from typing import (Any, Dict, List)

from . import debug_configs

__all__ = ['Operation', 'Cell']


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

    def to_forward_code(self, field: str, output: str, inputs: List[str]) -> str:
        from .converter.op_types import OpTypeName
        if self._to_class_name() is not None:
            return f'{output} = self.{field}({", ".join(inputs)})'
        elif self.type.startswith('Function.'):
            func_name = self.type[len('Function.'):]
            return f'{output} = F.{func_name}({", ".join(inputs)})'
        elif self.type == 'prim::Constant':
            if self.parameters:
                value = self.parameters['value']
            else:
                value = None
            return f'{output} = {value}'
        elif self.type == 'prim::ListConstruct':
            return f'{output} = [{", ".join(inputs)}]'
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
            assert len(inputs) == 2
            return f'{output} = {inputs[0]} + {inputs[1]}'
        elif self.type == OpTypeName.MergedSlice:
            assert (len(inputs) - 1) % 4 == 0
            slices = []
            dim = int((len(inputs) - 1) / 4)
            for i in range(dim):
                slices.append(f'{inputs[i*4+2]}:{inputs[i*4+3]}:{inputs[i*4+4]}')
            slice_str = ','.join(slices)
            return f'{output} = {inputs[0]}[{slice_str}]'
        elif self.type == 'aten::size':
            assert len(inputs) == 2
            return f'{output} = {inputs[0]}.size({inputs[1]})'
        elif self.type == 'aten::view':
            assert len(inputs) == 2
            return f'{output} = {inputs[0]}.view({inputs[1]})'
        elif self.type == 'aten::slice':
            raise RuntimeError('not supposed to have aten::slice operation')
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
