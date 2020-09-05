from . import utils


FuncOPs = ['aten::view', 'aten::relu', 'aten::log_softmax', 'aten::max_pool2d',
           'aten::cat', 'aten::size', 'aten::Int', 'aten::contiguous',
           'aten::mean', 'aten::add',
           'aten::transpose', 'aten::stack', 'aten::adaptive_avg_pool2d',
           'aten::avg_pool2d', 'aten::dropout', 'aten::slice']

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
        elif type in globals():
            OpClass = globals()[type]
        else:
            OpClass = AnyOperation
        return OpClass(type, _from_factory=True, **parameters)

    @staticmethod
    def load(data: 'Dict[str, Any]') -> 'Operation':
        return Operation.new(**data)

    def dump(self) -> 'Dict[str, Any]':
        data = dict(self.params)
        data['type'] = self.type
        return data

    def __repr__(self):
        return 'type: {}, params: {}'.format(self.type, self.params)

class FuncOp(Operation):
    # no need to generate code for functional ops here
    def to_tensorflow_init(self):
        return None

    def to_pytorch_init(self):
        return None

class Identity(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Identity()'

class Conv1d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv1d({parameters})'

class Conv2d(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.Conv2D(filters={}, kernel_size={}, padding="same", activation={})'.format(
            self.params['filters'],
            self.params['kernel_size'],
            '"{}"'.format(self.params['activation']) if 'activation' in self.params else 'None'
        )

    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv2d({parameters})'

class MaxPool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.MaxPool2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )

    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.MaxPool2d({parameters})'

MaxPool2d = MaxPool2D

class ZeroPad2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.ZeroPad2d({parameters})'

class AdaptiveAvgPool2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.AdaptiveAvgPool2d({parameters})'

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

class BatchNorm1d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.BatchNorm1d({parameters})'

class BatchNorm2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.BatchNorm2d({parameters})'

class AveragePool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.AvgPool2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )

class AvgPool2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.AvgPool2d({parameters})'

class Dropout(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Dropout({parameters})'

class ReLU(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.ReLU({parameters})'

class ReLU6(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.ReLU6({parameters})'

class LogSoftmax(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.LogSoftmax({parameters})'

class View(Operation):
    def to_pytorch_init(self):
        return 'CUSTOM.View(shape={})'.format(tuple(self.params.get('shape', (-1, ))))

class ShuffleNetBlock(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'ShuffleNetBlock({parameters})'

class LayerChoice(Operation):
    def to_pytorch_init(self):
        assert len(self.params) == 1
        k = 'op_candidates'
        v = self.params[k]
        assert isinstance(v, list)
        params_str = ', '.join(v)
        return 'mutables.LayerChoice([{}])'.format(params_str)

# for textnas specific op
class Mask(Operation):
    def to_pytorch_init(self):
        return 'Mask()'

class LinearCombine(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'LinearCombine({parameters})'

class GlobalAvgPool(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'GlobalAvgPool({parameters})'

class Attention(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'Attention({parameters})'

class WrapperOp(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'WrapperOp({parameters})'

class RNN(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'RNN({parameters})'

# for nasnet specific op
class Cell(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'Cell({parameters})'

class CellStem0(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CellStem0({parameters})'

class CellStem1(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CellStem1({parameters})'

# proxylessnas specific
class ProxylessNASMixedOp(Operation):
    def to_pytorch_init(self):
        operations = [Operation.load(op) for op in self.params['operations']]
        parameters = ', '.join([op.to_pytorch_init() for op in operations])
        return f'ProxylessNASMixedOp([{parameters}])'

# placeholder
class Placeholder(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'Placeholder({parameters})'

# mnasnet mutators
class RegularConv(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'RegularConv({parameters})'

class DepthwiseConv(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'DepthwiseConv({parameters})'

class MobileConv(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'MobileConv({parameters})'

class AnyOperation(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join([f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'{self.type}({parameters})'
