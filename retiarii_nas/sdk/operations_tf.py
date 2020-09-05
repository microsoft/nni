from typing import (Any, Dict)

from . import utils


FuncOPs = ['aten::view', 'aten::relu', 'aten::log_softmax', 'aten::max_pool2d', 'aten::cat']

class Operation:
    def __init__(self, type: str, _from_factory: bool = False, **parameters) -> None:
        assert _from_factory, 'Operation object can only be created with `Operation.new()`'
        self.type = type
        self.params = parameters

    def to_tensorflow_init(self) -> str:
        raise NotImplementedError()

    def to_pytorch_init(self) -> str:
        raise NotImplementedError()

    def update_params(self, **parameters) -> None:
        self.params.update(parameters)

    @staticmethod
    def new(type: str, **parameters) -> 'Operation':
        if type in FuncOPs:
            OpClass = globals()['FuncOp']
        else:
            OpClass = globals()[type]
        return OpClass(type, _from_factory=True, **parameters)

    @staticmethod
    def load(data: Dict[str, Any]) -> 'Operation':
        return Operation.new(**data)

    def dump(self) -> Dict[str, Any]:
        return {'type': self.type, **self.params}

class FuncOp(Operation):
    # no need to generate code for functional ops here
    def to_tensorflow_init(self):
        return None

    def to_pytorch_init(self):
        return None

class Identity(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Identity()'

class Conv2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.Conv2D(filters={}, kernel_size={}, padding="same", activation={})'.format(
            self.params['filters'],
            self.params['kernel_size'],
            '"{}"'.format(self.params['activation']) if 'activation' in self.params else 'None'
        )

    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv2d({parameters})'

class DepthwiseConv2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.DepthwiseConv2D(kernel_size={}, padding="same", activation={})'.format(
            self.params['kernel_size'],
            '"{}"'.format(self.params['activation']) if 'activation' in self.params else 'None'
        )

class MaxPool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.MaxPool2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )

    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.MaxPool2d({parameters})'

class Flatten(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.Flatten()'

class Linear(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.Dense(units={})'.format(
            self.params['units']
        )

    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Linear({parameters})'

class BatchNormalization(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.BatchNormalization()'

class AveragePool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.AveragePooling2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )

class ReLU(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.ReLU()'

    def to_pytorch_init(self):
        return 'nn.ReLU()'

class LogSoftmax(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.LogSoftmax({parameters})'

class View(Operation):
    def to_pytorch_init(self):
        return 'CUSTOM.View(shape={})'.format(tuple(self.params.get('shape', (-1, ))))

class Concatenate(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Concat({})'.format(self.params['dimension'])

class Split(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Split({})'.format(self.params['dimension'])

class Sum(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Sum()'

class Replication(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Replication()'


# ad-hoc ops for simplicity

class Hierarchical__none(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__none()'

class Hierarchical__1x1_conv(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__1x1_conv()'

class Hierarchical__3x3_depth_conv(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__3x3_depth_conv()'

class Hierarchical__3x3_sep_conv(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__3x3_sep_conv()'

class Hierarchical__3x3_max_pool(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__3x3_max_poll()'

class Hierarchical__3x3_avg_pool(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Hierarchical__3x3_avg_pool()'

class Wann__activation(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Wann__activation()'

class Wann__zero(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Wann__zero()'

class PathLevel__split(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.PathLevel__split()'

class PathLevel__avg_pool(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.PathLevel__avg_pool()'

class PathLevel__max_pool(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.PathLevel__max_pool()'
