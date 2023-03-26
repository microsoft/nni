from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import nni
from nni.nas.nn.pytorch import (
    ModelSpace, Cell, ParametrizedModule, LayerChoice, InputChoice, MutableLinear,
    MutableMultiheadAttention, MutableConv2d, MutableBatchNorm2d, Repeat
)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SimpleNet(ModelSpace):
    def __init__(self, value_choice=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = LayerChoice([
            nn.Conv2d(32, 64, 3, 1),
            DepthwiseSeparableConv(32, 64)
        ], label='conv')
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ], label='dropout')
        self.dropout2 = nn.Dropout(0.5)
        if value_choice:
            hidden = nni.choice('hidden', [32, 64, 128])
        else:
            hidden = 64
        self.fc1 = MutableLinear(9216, hidden)
        self.fc2 = MutableLinear(hidden, 10)
        self.rpfc = nn.Linear(10, 10)
        self.input_ch = InputChoice(2, 1, label='input')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x1 = self.rpfc(x)
        x = self.input_ch([x, x1])
        output = F.log_softmax(x, dim=1)
        return output


class MultiHeadAttentionNet(ModelSpace):
    def __init__(self, head_count):
        super().__init__()
        embed_dim = nni.choice('embed_dim', [32, 64])
        self.linear1 = MutableLinear(128, embed_dim)
        self.mhatt = MutableMultiheadAttention(embed_dim, head_count)
        self.linear2 = MutableLinear(embed_dim, 1)

    def forward(self, batch):
        query, key, value = batch
        q, k, v = self.linear1(query), self.linear1(key), self.linear1(value)
        output, _ = self.mhatt(q, k, v, need_weights=False)
        y = self.linear2(output)
        return F.relu(y)


class ValueChoiceConvNet(ModelSpace):
    def __init__(self):
        super().__init__()
        ch1 = nni.choice('ch1', [16, 32])
        kernel = nni.choice('kernel', [3, 5])
        self.conv1 = MutableConv2d(1, ch1, kernel, padding=kernel // 2)
        self.batch_norm = MutableBatchNorm2d(ch1)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = MutableConv2d(ch1, 64, 3)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.maxpool(self.conv2(x))
        x = torch.mean(x, (2, 3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class RepeatNet(ModelSpace):
    def __init__(self):
        super().__init__()
        ch1 = nni.choice('ch1', [16, 32])
        kernel = nni.choice('kernel', [3, 5])
        self.conv1 = MutableConv2d(1, ch1, kernel, padding=kernel // 2)
        self.batch_norm = MutableBatchNorm2d(ch1)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = MutableConv2d(ch1, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
        self.rpfc = Repeat(nn.Linear(10, 10), (1, 4), label='rep')

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.maxpool(self.conv2(x))
        x = torch.mean(x, (2, 3))
        x = self.fc(x)
        x = self.rpfc(x)
        return F.log_softmax(x, dim=1)


class CellNet(ModelSpace):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 5, 7, stride=4)
        self.cells = Repeat(
            lambda index: Cell({
                'conv1': lambda _, __, inp: nn.Conv2d(
                    (5 if index == 0 else 3 * 4) if inp is not None and inp < 1 else 4, 4, 1
                ),
                'conv2': lambda _, __, inp: nn.Conv2d(
                    (5 if index == 0 else 3 * 4) if inp is not None and inp < 1 else 4, 4, 3, padding=1
                ),
            }, 3, merge_op='loose_end', label=f'cell{index}'),
            (1, 3), label='rep'
        )
        self.fc = nn.Linear(3 * 4, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.cells(x)
        x = torch.mean(x, (2, 3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MyOp(ParametrizedModule):
    def __init__(self, some_ch):
        super().__init__()
        self.some_ch = some_ch
        self.batch_norm = nn.BatchNorm2d(some_ch)

    def forward(self, x):
        return self.batch_norm(x)


class CustomOpValueChoiceNet(ModelSpace):
    def __init__(self):
        super().__init__()
        ch1 = nni.choice('ch1', [16, 32])
        kernel = nni.choice('kernel', [3, 5])
        self.conv1 = MutableConv2d(1, ch1, kernel, padding=kernel // 2)
        self.batch_norm = MyOp(ch1)
        self.conv2 = MutableConv2d(ch1, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = LayerChoice([
            nn.Dropout(.25),
            nn.Dropout(.5),
            nn.Dropout(.75)
        ], label='dropout')
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.maxpool(self.conv2(x))
        x = torch.mean(x, (2, 3))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CellSimple(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                         num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell(x, y)


class CellDefaultArgs(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)], num_nodes=4)

    def forward(self, x):
        return self.cell(x)


class CellPreprocessor(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 16)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.linear(x[0]), x[1]]


class CellPostprocessor(nn.Module):
    def forward(self, this: torch.Tensor, prev: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return prev[-1], this


class CellCustomProcessor(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.cell = Cell({
            'first': nn.Linear(16, 16),
            'second': nn.Linear(16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2,
            preprocessor=CellPreprocessor(), postprocessor=CellPostprocessor(), merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])


class CellLooseEnd(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.cell = Cell([nn.Linear(16, 16), nn.Linear(16, 16, bias=False)],
                         num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='loose_end')

    def forward(self, x, y):
        return self.cell([x, y])


class CellOpFactory(ModelSpace, label_prefix='model'):
    def __init__(self):
        super().__init__()
        self.cell = Cell({
            'first': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16),
            'second': lambda _, __, chosen: nn.Linear(3 if chosen == 0 else 16, 16, bias=False)
        }, num_nodes=4, num_ops_per_node=2, num_predecessors=2, merge_op='all')

    def forward(self, x, y):
        return self.cell([x, y])


MODELS = {
    'simple': partial(SimpleNet, False),
    'simple_value_choice': SimpleNet,
    'value_choice': ValueChoiceConvNet,
    'repeat': RepeatNet,
    'cell': CellNet,
    'multihead_attention': partial(MultiHeadAttentionNet, 1),
    'custom_op': CustomOpValueChoiceNet,
    'cell_simple': CellSimple,
    'cell_default_args': CellDefaultArgs,
    'cell_custom_processor': CellCustomProcessor,
    'cell_loose_end': CellLooseEnd,
    'cell_op_factory': CellOpFactory
}
