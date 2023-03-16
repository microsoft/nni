# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Operations used in graph-based engine.
"""

from typing import (Any, Dict, List, Optional, cast)

from nni.common.framework import get_default_framework


__all__ = ['Operation', 'Cell', 'PyTorchOperation', 'TensorFlowOperation']


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

    io_names: List[str] = []

    def __init__(self, type_name: str, parameters: Dict[str, Any] = {}, _internal: bool = False, attributes: Dict[str, Any] = {}):
        assert _internal, '`Operation()` is private, use `Operation.new()` instead'
        self.type: str = type_name
        self.parameters: Dict[str, Any] = parameters
        self.attributes: Dict[str, Any] = attributes

    def to_init_code(self, field: str) -> str:
        raise NotImplementedError()

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any]) -> str:
        raise NotImplementedError()

    def _to_class_name(self) -> str:
        raise NotImplementedError()

    def __bool__(self) -> bool:
        return True

    @staticmethod
    def new(type_name: str, parameters: Dict[str, Any] = cast(Dict[str, Any], None), cell_name: str = cast(str, None),
            attributes: Dict[str, Any] = cast(Dict[str, Any], None)) -> 'Operation':
        parameters = parameters or {}
        attributes = attributes or {}
        if type_name == '_cell':
            # NOTE: cell_name is the same as its Node's name, when the cell is wrapped within the node
            return Cell(cell_name, parameters)
        else:
            if get_default_framework() in ('torch', 'pytorch'):
                from nni.nas.space.pytorch import op_def  # pylint: disable=unused-import
                cls = PyTorchOperation._find_subclass(type_name)
            elif get_default_framework() in ('tf', 'tensorflow'):
                from nni.nas.space.tensorflow import op_def  # pylint: disable=unused-import
                cls = TensorFlowOperation._find_subclass(type_name)
            else:
                raise ValueError(f'Unsupported framework: {get_default_framework()}')
            return cls(type_name, parameters, _internal=True, attributes=attributes)

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
    @classmethod
    def _find_subclass(cls, subclass_name):
        if cls.to_class_name(subclass_name) is not None:
            subclass_name = 'ModuleOperator'
        if cls.is_functional(subclass_name):
            subclass_name = 'FunctionalOperator'
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '_ori_type_name') and \
                    subclass_name in cast(Any, subclass)._ori_type_name:
                return subclass
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '_artificial_op_name') and \
                    subclass_name in cast(Any, subclass)._artificial_op_name:
                return subclass
        return cls

    @classmethod
    def to_class_name(cls, type_name) -> Optional[str]:
        if type_name.startswith('__torch__.'):
            return type_name[len('__torch__.'):]
        elif type_name.startswith('__mutated__.'):
            return type_name[len('__mutated__.'):]
        else:
            return None

    @classmethod
    def is_functional(cls, type_name) -> bool:
        return type_name.startswith('Function.')

    def _to_class_name(self) -> Optional[str]:
        if self.type.startswith('__torch__.'):
            return self.type[len('__torch__.'):]
        elif self.type.startswith('__mutated__.'):
            return self.type[len('__mutated__.'):]
        else:
            return None

    def get_import_pkg(self) -> Optional[str]:
        if self.type.startswith('__torch__.'):
            return self.type[len('__torch__.'):].split('.')[0]
        elif self.type.startswith('__mutated__.'):
            return self.type[len('__mutated__.'):].split('.')[0]
        else:
            return None

    def to_init_code(self, field: str) -> Optional[str]:
        if self._to_class_name() is not None:
            assert 'positional_args' not in self.parameters
            kw_params = ', '.join(f'{key}={repr(value)}' for key, value in self.parameters.items())
            return f'self.{field} = {self._to_class_name()}({kw_params})'
        return None

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any]) -> str:
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

        Returns
        -------
        str
            generated code line
        """
        if self.type == 'aten::slice':
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

    def __init__(self, cell_name: str,
                 parameters: Dict[str, Any] = cast(Dict[str, Any], None),
                 attributes: Dict[str, Any] = cast(Dict[str, Any], None)):
        self.type = '_cell'
        self.cell_name = cell_name
        self.parameters = parameters or {}
        self.attributes = attributes or {}

    def _to_class_name(self):
        # TODO: ugly, think about how to refactor this part
        return _convert_name(self.cell_name)

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any]) -> str:
        return f'{output} = self.{field}({", ".join(inputs)})'


class _IOPseudoOperation(Operation):
    """
    This is the pseudo operation used by I/O nodes.
    The benefit is that users no longer need to verify `Node.operation is not None`,
    especially in static type checking.
    """

    def __init__(self, type_name: str, io_names: List[str] = cast(List[str], None)):
        assert type_name.startswith('_')
        super(_IOPseudoOperation, self).__init__(type_name, {}, True)
        self.io_names = io_names

    def to_init_code(self, field: str) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def to_forward_code(self, field: str, output: str, inputs: List[str], inputs_value: List[Any]) -> str:
        raise ValueError(f'Cannot generate code for pseudo operation "{self.type}"')

    def __bool__(self) -> bool:
        return False
