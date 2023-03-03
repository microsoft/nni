from typing import Type
from types import ModuleType

import pytest
import torch
import torch.nn

from nni.mutable import Categorical
from nni.nas.space import ExecutableModelSpace, GraphModelSpace
from nni.nas.nn.pytorch import Repeat, ModelSpace, LayerChoice

from .test_choice import space_format, nn

class AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_repeat(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(AddOne(), (3, 5))

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    label = list(model.simplify())[0]
    for t in [3, 4, 5]:
        selected_model = model.freeze({label: t}).executable_model()
        if not issubclass(space_format, GraphModelSpace):
            assert isinstance(list(selected_model.children())[0], nn.Sequential)
        assert (selected_model(torch.zeros(1, 16)) == t).all()



def test_repeat_static(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(lambda index: LayerChoice([AddOne(), nn.Identity()]), 4)

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 4
    result = []
    for _ in range(50):
        selected_model = model.random()
        result.append(selected_model.executable_model()(torch.zeros(1, 1)).item())
    for x in [1, 2, 3]:
        assert float(x) in result


def test_repeat_complex(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(LayerChoice([AddOne(), nn.Identity()], label='lc'), (3, 5), label='rep')

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    simplified = model.simplify()
    assert len(simplified) == 2

    model1 = model.freeze({'model/lc': simplified['model/lc'].values[0], 'model/rep': 3}).executable_model()
    model2 = model.freeze({'model/lc': simplified['model/lc'].values[1], 'model/rep': 4}).executable_model()
    model3 = model.freeze({'model/lc': simplified['model/lc'].values[0], 'model/rep': 5}).executable_model()

    assert model1(torch.zeros(1, 1)).item() == 3
    assert model2(torch.zeros(1, 1)).item() == 0
    assert model3(torch.zeros(1, 1)).item() == 5


def test_repeat_complex_independent(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(lambda index: LayerChoice([AddOne(), nn.Identity()]), (2, 3), label='rep')

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 4

    result = []
    for selected_model in model.grid():
        selected_model = selected_model.executable_model()
        result.append(selected_model(torch.zeros(1, 1)).item())
    assert sorted(result) == sorted([0, 1, 1, 1, 2, 2, 2, 3] + [0, 0, 1, 1, 1, 1, 2, 2])


def test_repeat_discrete(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(AddOne(), Categorical([1, 3, 5], label='ds'))

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    for target in [1, 3, 5]:
        selected_model = model.freeze({'model/ds': target}).executable_model()
        assert (selected_model(torch.zeros(1, 16)) == target).all()


def test_repeat_mutable_expr(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(AddOne(), Categorical([0, 2, 4]) + 1)

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    for target, selected_model in zip([1, 3, 5], model.grid()):
        selected_model = selected_model.executable_model()
        assert (selected_model(torch.zeros(1, 16)) == target).all()


def test_repeat_zero(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.block = Repeat(AddOne(), (0, 3))

        def forward(self, x):
            return self.block(x)

    model = space_format.from_model(Net())
    if issubclass(space_format, GraphModelSpace):
        with pytest.raises(AssertionError):
            model.simplify()
    else:
        assert len(model.simplify()) == 1
        label = list(model.simplify())[0]
        for target in [0, 1, 2, 3]:
            new_model = model.freeze({label: target}).executable_model()
            assert (new_model(torch.zeros(1, 16)) == target).all()


def test_repeat_contains():
    import torch.nn as nn

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.repeat = Repeat(
                lambda index: LayerChoice([nn.Identity(), nn.Identity()], label=f'layer{index}'),
                (3, 5), label='rep')

        def forward(self, x):
            return self.module(x)

    net = Net()
    assert net.contains({'rep': 3, 'layer0': 0, 'layer1': 0, 'layer2': 0})
    assert not net.contains({'rep': 4, 'layer0': 0, 'layer1': 0, 'layer2': 0})
    assert net.contains({'rep': 3, 'layer0': 0, 'layer1': 0, 'layer2': 0, 'layer3': 0})
