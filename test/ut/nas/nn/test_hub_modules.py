from types import ModuleType
from typing import Type

import pytest
import torch
from torch import nn

from nni.mutable import ConstraintViolation
from nni.nas.hub.pytorch.modules import AutoActivation, NasBench101Cell, NasBench201Cell
from nni.nas.nn.pytorch import ModelSpace
from nni.nas.space import ExecutableModelSpace, GraphModelSpace

from .test_choice import space_format, nn


def test_nasbench201_cell(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.cell = NasBench201Cell([
                lambda x, y: nn.Linear(x, y),
                lambda x, y: nn.Linear(x, y, bias=False)
            ], 10, 16, label='cell1')

        def forward(self, x):
            return self.cell(x)

    net = Net()
    assert net.cell.label == 'cell1'
    model = space_format.from_model(net)
    assert len(model.simplify()) == 6
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 16])


def test_autoactivation(space_format: Type[ExecutableModelSpace]):
    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.act = AutoActivation(unit_num=2, label='abc')
            assert self.act.label == 'abc'

        def forward(self, x):
            return self.act(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 5
    assert set(model.simplify().keys()) == set([
        'abc/unary_0', 'abc/unary_1', 'abc/unary_2', 'abc/binary_0', 'abc/binary_1'
    ])
    for _ in range(10):
        selected_model = model.random().executable_model()
        assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 10])


def test_nasbench101_cell(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    if issubclass(space_format, GraphModelSpace):
        pytest.skip('GraphSpace does not support NasBench101Cell')

    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.cell = NasBench101Cell([lambda x: nn.Linear(x, x), lambda x: nn.Linear(x, x, bias=False)],
                                        10, 16, lambda x, y: nn.Linear(x, y), max_num_nodes=5, max_num_edges=7)

        def forward(self, x):
            return self.cell(x)

    net = Net()
    assert net.cell.label == 'model/1'
    model = space_format.from_model(net)
    simplified = model.simplify()
    expected_keys = ['model/1/num_nodes'] + [f'model/1/op{i}' for i in range(1, 4)] + [f'model/1/input{i}' for i in range(1, 5)] + ['model/1/final']
    assert set(simplified.keys()) == set(expected_keys)

    succeed_count = 0
    for _ in range(30):
        try:
            selected_model = model.random().executable_model()
            assert selected_model(torch.randn(2, 10)).size() == torch.Size([2, 16])
            succeed_count += 1
        except ConstraintViolation as e:
            assert 'at most' in str(e) or 'less than' in str(e)
    assert 0 < succeed_count < 30
