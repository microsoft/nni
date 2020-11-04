from typing import (Any, Dict)

from . import debug_configs

__all__ = ['Operation', 'Cell']


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
        params = ', '.join(f'{key}={repr(value)}' for key, value in self.parameters.items())
        return f'self.{field} = {self._to_class_name()}({params})'

    def to_forward_code(self, field: str, output: str, *inputs: str) -> str:
        return f'{output} = self.{field}({", ".join(inputs)})'

    def _to_class_name(self) -> str:
        raise NotImplementedError()

    def __bool__(self) -> bool:
        return True

    @staticmethod
    def new(type_name: str, parameters: Dict[str, Any] = {}) -> 'Operation':
        if type_name == '_cell':
            return Cell(parameters['cell'])
        else:
            if debug_configs.framework.lower() in ('torch', 'pytorch'):
                from .operation_def import torch_op_def  # pylint: disable=unused-import
                cls = PyTorchOperation._find_subclass(type)
            elif debug_configs.framework.lower() in ('tf', 'tensorflow'):
                from .operation_def import tf_op_def  # pylint: disable=unused-import
                cls = TensorFlowOperation._find_subclass(type)
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
        return 'nn.' + self.type

class TensorFlowOperation(Operation):
    def _to_class_name(self) -> str:
        return 'K.layers.' + self.type


class Cell(Operation):
    """
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
        return self.cell_name


class _PseudoOperation(Operation):
    """
    This is the pseudo operation used by I/O nodes.
    The benefit is that users no longer need to verify `Node.operation is not None`,
    especially in static type checking.
    """
    def __init__(self, type_name: str):
        assert type_name.startswith('_')
        self.type = type_name
        self.parameters = {}

    def to_init_code(self, field: str) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def to_forward_code(self, field: str, output: str, *inputs: str) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def __bool__(self) -> bool:
        return False
