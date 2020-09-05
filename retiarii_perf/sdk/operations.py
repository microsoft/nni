from . import utils


FuncOPs = ['aten::view', 'aten::relu', 'aten::log_softmax', 'aten::max_pool2d',
           'aten::cat', 'aten::size', 'aten::Int', 'aten::contiguous',
           'aten::mean', 'aten::add', 'aten::transpose', 'aten::stack',
           'aten::avg_pool2d', 'aten::dropout']


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
    def load(data: 'Dict[str, Any]') -> 'Operation':
        return Operation.new(**data)

    def dump(self) -> 'Dict[str, Any]':
        data = dict(self.params)
        data['type'] = self.type
        return data


class FuncOp(Operation):
    # no need to generate code for functional ops here
    def to_tensorflow_init(self):
        return None

    def to_pytorch_init(self):
        return None


class DataLoader(Operation):
    def to_pytorch_init(self):
        return "DataLoaderTODO"


class LossFunction(Operation):
    def to_pytorch_init(self):
        return "LossFunctionTODO"


class Identity(Operation):
    def to_tensorflow_init(self):
        return 'CUSTOM.Identity()'

    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Identity({parameters})'


class Conv1d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv1d({parameters})'


class Conv2d(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.Conv2D(filters={}, kernel_size={}, padding="same", activation={})'.format(
            self.params['filters'],
            self.params['kernel_size'],
            '"{}"'.format(self.params['activation']
                          ) if 'activation' in self.params else 'None'
        )

    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv2d({parameters})'


class Conv1d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Conv1d({parameters})'


class Dropout(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Dropout({parameters})'


class MaxPool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.MaxPool2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )

    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
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
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Linear({parameters})'


class BatchNormalization(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.BatchNormalization()'


class BatchNorm1d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.BatchNorm1d({parameters})'


class BatchNorm2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.BatchNorm2d({parameters})'


class AveragePool2D(Operation):
    def to_tensorflow_init(self):
        return 'K.layers.AvgPool2D(pool_size={}, padding="same")'.format(
            self.params['pool_size']
        )


class AvgPool2d(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.AvgPool2d({parameters})'


class Dropout(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.Dropout({parameters})'


class ReLU(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.ReLU({parameters})'

class FixedInputChoice(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CUSTOM.FixedInputChoice({parameters}, gen=True)'

class LogSoftmax(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'nn.LogSoftmax({parameters})'


class View(Operation):
    def to_pytorch_init(self):
        return 'CUSTOM.View(shape={})'.format(tuple(self.params.get('shape', (-1, ))))


class Broadcast(Operation):
    def to_pytorch_init(self):
        if self.params["is_src"]:
            return f'CUSTOM.Broadcast(True, {self.params["rank"]})'
        else:
            return f'CUSTOM.Broadcast(False, {self.params["rank"]}, size = {self.params["size"]}, dtype = {self.params["dtype"]}, device = {self.params["device"]})'


class SuperGraphOpChoices(Operation):
    def to_pytorch_init(self):
        choice_str = ",".join([Operation.load(_).to_pytorch_init()
                               for _ in self.params["candidates"]])
        return f'mutables.LayerChoice([{choice_str}])'


class BertEmbedding(Operation):
    def to_pytorch_init(self):
        return f'CUSTOM.Bert("{self.params["pretrain_model"]}", {self.params["max_length"]}, {self.params["pad_to_max_length"]}, {self.params["truncation"]})'


class Transpose(Operation):
    def to_pytorch_init(self):
        return f'CUSTOM.Transpose(dim0={self.params["dim0"]}, dim1={self.params["dim1"]})'


class Squeeze(Operation):
    def to_pytorch_init(self):
        return f'CUSTOM.Squeeze()'


class Expand(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CUSTOM.Expand({parameters})'

class BatchSizeView(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CUSTOM.BatchSizeView({parameters})'

class ShuffleNetBlock(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
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


class Bool(Operation):
    def to_pytorch_init(self):
        return 'Bool()'


class LinearCombine(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'LinearCombine({parameters})'


class GlobalAvgPool(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'GlobalAvgPool({parameters})'


class Attention(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
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
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'Cell({parameters})'


class CellStem0(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CellStem0({parameters})'


class CellStem1(Operation):
    def to_pytorch_init(self):
        parameters = ', '.join(
            [f'{k}={repr(v)}' for k, v in self.params.items()])
        return f'CellStem1({parameters})'
