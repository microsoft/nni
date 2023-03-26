import copy
import random
from types import ModuleType
from typing import Type

import pytest

import nni
import nni.nas.nn.pytorch.layers
import nni.nas.evaluator.pytorch.lightning as pl
import torch
import torch.nn.functional as F

from nni.mutable import Categorical, ensure_frozen, label_scope, SampleValidationError
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.space import (
    RawFormatModelSpace, SimplifiedModelSpace,
    ExecutableModelSpace, GraphModelSpace,
    model_context
)
from nni.nas.space.pytorch import PytorchGraphModelSpace
from nni.nas.nn.pytorch import (
    ParametrizedModule, MutableModule, ModelSpace,
    LayerChoice, InputChoice, Cell,
    ChosenInputs, MutableConv2d, MutableLinear,
)


@pytest.fixture(params=['raw', 'simple', 'graph'])
def space_format(request):
    if request.param == 'raw':
        return RawFormatModelSpace
    elif request.param == 'simple':
        return SimplifiedModelSpace
    elif request.param == 'graph':
        return PytorchGraphModelSpace
    else:
        raise ValueError(f'Unknown space format: {request.param}')


@pytest.fixture
def nn(space_format: Type[ExecutableModelSpace]):
    if space_format == PytorchGraphModelSpace:
        return nni.nas.nn.pytorch.layers
    else:
        return torch.nn


def test_simplify_freeze():
    from torch import nn

    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([
                nn.Conv2d(3, 3, kernel_size=1),
                nn.Conv2d(3, 5, kernel_size=1)
            ])

        def forward(self, x):
            return self.module(x)

    model = Net()
    assert model.module.names == [0, 1]
    assert model.module.label == 'model/1'
    assert len(model.module) == 2
    space = model.simplify()
    assert len(space) == 1
    assert isinstance(next(iter(space.values())), Categorical)

    layer_label = next(iter(space.keys()))
    assert space[layer_label].values == [0, 1]

    model1 = model.freeze({layer_label: 0})
    model2 = model.freeze({layer_label: 1})

    assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 3, 3, 3])
    assert model2(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 5, 3, 3])


def test_layer_choice(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([
                nn.Conv2d(3, 2, kernel_size=1),
                nn.Conv2d(3, 5, kernel_size=1)
            ])

        def forward(self, x):
            return self.module(x)

    model = Net()
    with pytest.raises(KeyError, match='allow adding'):
        model.module[2] = nn.Conv2d(3, 7, kernel_size=1)

    model.module[0] = nn.Conv2d(3, 3, kernel_size=1)

    assert model.module[1].out_channels == 5

    model = space_format.from_model(model)

    space = model.simplify()
    assert len(space) == 1
    assert isinstance(next(iter(space.values())), Categorical)

    layer_label = next(iter(space.keys()))
    choices = space[layer_label].values

    model1 = model.freeze({layer_label: choices[0]}).executable_model()
    model2 = model.freeze({layer_label: choices[1]}).executable_model()

    if space_format == SimplifiedModelSpace:
        # replace doesn't work in simplified space
        assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 2, 3, 3])
    else:
        assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 3, 3, 3])
    assert model2(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 5, 3, 3])


def test_layer_choice_dict(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice({
                'a': nn.Conv2d(3, 2, kernel_size=1),
                'b': nn.Conv2d(3, 5, kernel_size=1)
            }, label='x')

        def forward(self, x):
            return self.module(x)

    model = Net()
    with pytest.raises(KeyError, match='allow adding'):
        model.module['c'] = nn.Conv2d(3, 7, kernel_size=1)
    model.module['a'] = nn.Conv2d(3, 3, kernel_size=1)

    assert model.module['b'].out_channels == 5

    model = space_format.from_model(model)

    space = model.simplify()
    if issubclass(space_format, GraphModelSpace):
        # The choice in GraphModelSpace can be quite unexpected.
        choices = space['x'].values
    else:
        choices = ['a', 'b']
        assert space['x'].values == choices

    model1 = model.freeze({'x': choices[0]}).executable_model()
    model2 = model.freeze({'x': choices[1]}).executable_model()

    if space_format == SimplifiedModelSpace:
        assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 2, 3, 3])
    else:
        assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 3, 3, 3])
    assert model2(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 5, 3, 3])


def test_layer_choice_multiple(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([nn.Conv2d(3, i, kernel_size=1) for i in range(1, 11)])

        def forward(self, x):
            return self.module(x)

    model = Net()
    model = space_format.from_model(model)
    space = model.simplify()
    assert len(space) == 1
    layer_label = next(iter(space.keys()))
    choices = space[layer_label].values

    for i in range(1, 11):
        model_new = model.freeze({layer_label: choices[i - 1]}).executable_model()
        assert model_new(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, i, 3, 3])


def test_nested_layer_choice(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([
                LayerChoice([nn.Conv2d(3, 3, kernel_size=1),
                             nn.Conv2d(3, 4, kernel_size=1),
                             nn.Conv2d(3, 5, kernel_size=1)]),
                nn.Conv2d(3, 1, kernel_size=1)
            ])

        def forward(self, x):
            return self.module(x)

    model = Net()
    model = space_format.from_model(model)
    assert len(model.simplify()) == 2

    expect_shape_list = [1, 1, 1, 3, 4, 5]
    shape_list = []

    for new_model in model.grid():
        shape_list.append(new_model.executable_model()(torch.randn(1, 3, 5, 5)).size(1))
    assert sorted(shape_list) == expect_shape_list


def test_input_choice(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv2 = nn.Conv2d(3, 5, kernel_size=1)
            self.input = InputChoice(2)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            return self.input([x1, x2])

    model = Net()
    assert model.input.label == 'model/1'
    model = space_format.from_model(model)
    assert len(model.simplify()) == 1
    label = next(iter(model.simplify().keys()))

    if issubclass(space_format, GraphModelSpace):
        # Graph space doesn't support list
        model1 = model.freeze({label: 0}).executable_model()
        model2 = model.freeze({label: 1}).executable_model()
    else:
        model1 = model.freeze({label: [0]}).executable_model()
        model2 = model.freeze({label: [1]}).executable_model()

    assert model1(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 3, 3, 3])
    assert model2(torch.randn(1, 3, 3, 3)).size() == torch.Size([1, 5, 3, 3])


def test_chosen_inputs(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self, reduction):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv2 = nn.Conv2d(3, 3, kernel_size=1)
            self.input = InputChoice(2, n_chosen=2, reduction=reduction)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            return self.input([x1, x2])

    for reduction in ['none', 'sum', 'mean', 'concat']:
        model = space_format.from_model(Net(reduction))
        selected_model = model.random().executable_model()
        result = selected_model(torch.randn(1, 3, 3, 3))
        if reduction == 'none':
            assert len(result) == 2
            assert result[0].size() == torch.Size([1, 3, 3, 3])
            assert result[1].size() == torch.Size([1, 3, 3, 3])
        elif reduction == 'concat':
            assert result.size() == torch.Size([1, 6, 3, 3])
        else:
            assert result.size() == torch.Size([1, 3, 3, 3])


def test_discrete_as_parameter(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv2d(3, 5, kernel_size=Categorical([3, 5], label='kz'))

        def forward(self, x):
            return self.conv(x)

    model = space_format.from_model(Net())
    with label_scope('model'):
        mutable_ref = Categorical([3, 5], label='kz')
    assert mutable_ref.equals(model.simplify()['model/kz'])
    model1 = model.freeze({'model/kz': 3}).executable_model()
    model2 = model.freeze({'model/kz': 5}).executable_model()
    if not issubclass(space_format, GraphModelSpace):
        assert type(list(model1.children())[0]) is nn.Conv2d
    assert model1(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 5, 3, 3])
    assert model2(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 5, 1, 1])


def test_discrete_tuple_warning(caplog):
    from torch import nn

    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv2d(3, 5, kernel_size=(Categorical([3, 5]), Categorical([3, 5])))

        def forward(self, x):
            return self.conv(x)

    with pytest.raises(TypeError):
        Net()
    assert 'nested mutable' in caplog.text


def test_value_choice_as_two_parameters(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv2d(
                3,
                Categorical([6, 8]),
                kernel_size=Categorical([3, 5])
            )

        def forward(self, x):
            return self.conv(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 2
    all_shapes = []
    for selected_model in model.grid():
        shape = selected_model.executable_model()(torch.randn(1, 3, 5, 5)).size()
        all_shapes.append((shape[1], shape[2]))
    assert set(all_shapes) == set([(6, 3), (6, 1), (8, 3), (8, 1)])


def test_value_choice_as_parameter_shared(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='net'):
        def __init__(self):
            super().__init__()
            self.conv1 = MutableConv2d(3, Categorical([6, 8], label='shared'), 1)
            self.conv2 = MutableConv2d(3, Categorical([6, 8], label='shared'), 1)

        def forward(self, x):
            return self.conv1(x) + self.conv2(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    model1 = model.freeze({'net/shared': 6}).executable_model()
    model2 = model.freeze({'net/shared': 8}).executable_model()
    assert model1(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 6, 5, 5])
    assert model2(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 8, 5, 5])


def test_value_choice_backward_compatibility(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    # Backward compatibility for v2.x
    class Net(ModelSpace, label_prefix=None):
        def __init__(self):
            super().__init__()
            with pytest.deprecated_call():
                vc = nni.nas.nn.pytorch.ValueChoice([3, 5], label='kz')
            self.conv = nni.nas.nn.pytorch.layers.Conv2d(3, 5, kernel_size=vc)

        def forward(self, x):
            return self.conv(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    model1 = model.freeze({'kz': 3}).executable_model()
    assert model1(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 5, 3, 3])


def test_value_choice_in_functional(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    # We no longer support this.
    # Testing it to make sure it raises an error.

    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.dropout_rate = Categorical([0., 1.])

        def forward(self, x):
            return F.dropout(x, self.dropout_rate())

    if issubclass(space_format, GraphModelSpace):
        with pytest.raises(RuntimeError, match='TorchScript'):
            model = space_format.from_model(Net())
    else:
        model = space_format.from_model(Net())
        assert len(model.simplify()) == 0
        with pytest.raises(TypeError, match='not callable'):
            model.freeze({}).executable_model()(torch.randn(1, 3, 3, 3))


def test_value_choice_in_layer_choice(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.linear = LayerChoice([
                MutableLinear(3, Categorical([10, 20])),
                MutableLinear(3, Categorical([30, 40]))
            ])

        def forward(self, x):
            return self.linear(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 3
    sizes = []
    for selected_model in model.grid():
        sizes.append(selected_model.executable_model()(torch.randn(1, 3)).size(1))
    assert len(sizes) == 8  # 2 * 2 * 2
    assert set(sizes) == set([10, 20, 30, 40])


def test_shared(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self, shared=True):
            super().__init__()
            labels = ['x', 'x'] if shared else [None, None]
            self.module1 = LayerChoice([
                MutableConv2d(3, 3, kernel_size=1),
                MutableConv2d(3, 5, kernel_size=1)
            ], label=labels[0])
            self.module2 = LayerChoice([
                MutableConv2d(3, 3, kernel_size=1),
                MutableConv2d(3, 5, kernel_size=1)
            ], label=labels[1])

        def forward(self, x):
            return self.module1(x) + self.module2(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    # sanity check
    assert model.random().executable_model()(torch.randn(1, 3, 3, 3)).size(0) == 1

    model = space_format.from_model(Net(shared=False))
    assert len(model.simplify()) == 2
    # repeat test. Expectation: sometimes succeeds, sometimes fails.
    failed_count = 0
    for _ in range(30):
        try:
            model.random().executable_model()(torch.randn(1, 3, 3, 3))
        except RuntimeError:
            failed_count += 1
    assert 0 < failed_count < 30


def test_discrete_getitem(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix=None):
        def __init__(self):
            super().__init__()
            vc = Categorical([(6, 3), (8, 5)], label='vc')
            self.conv = MutableConv2d(3, vc[0], kernel_size=vc[1])

        def forward(self, x):
            return self.conv(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    model1 = model.freeze({'vc': (6, 3)}).executable_model()
    model2 = model.freeze({'vc': (8, 5)}).executable_model()
    input = torch.randn(1, 3, 5, 5)
    assert model1(input).size() == torch.Size([1, 6, 3, 3])
    assert model2(input).size() == torch.Size([1, 8, 1, 1])


def test_discrete_getitem_dict(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            choices = [
                {'b': [3], 'bp': [6]},
                {'b': [6], 'bp': [12]}
            ]
            self.conv = MutableConv2d(3, Categorical(choices, label='a')['b'][0], 1)
            self.conv1 = MutableConv2d(Categorical(choices, label='a')['bp'][0], 3, 1)

        def forward(self, x):
            x = self.conv(x)
            return self.conv1(torch.cat((x, x), 1))

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 1
    input = torch.randn(1, 3, 5, 5)
    # Sanity check
    model.random().executable_model()(input)


def test_discrete_multi(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            choice1 = Categorical([{"in": 1, "out": 3}, {"in": 2, "out": 6}, {"in": 3, "out": 9}])
            choice2 = Categorical([2.5, 3.0, 3.5], label='multi')
            choice3 = Categorical([2.5, 3.0, 3.5], label='multi')
            self.conv1 = MutableConv2d(choice1["in"], round(choice1["out"] * choice2), 1)
            self.conv2 = MutableConv2d(choice1["in"], round(choice1["out"] * choice3), 1)

        def forward(self, x):
            return self.conv1(x) + self.conv2(x)

    model = space_format.from_model(Net())
    assert len(model.simplify()) == 2

    for i, selected_model in enumerate(model.grid()):
        input = torch.randn(1, i // 3 + 1, 3, 3)
        expected_shape = torch.Size([1, round((i // 3 + 1) * 3 * [2.5, 3.0, 3.5][i % 3]), 3, 3])
        assert selected_model.executable_model()(input).shape == expected_shape


def test_discrete_inconsistent_label():
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv1 = MutableConv2d(3, Categorical([3, 5], label='a'), 1)
            self.conv2 = MutableConv2d(3, Categorical([3, 6], label='a'), 1)

        def forward(self, x):
            return torch.cat([self.conv1(x), self.conv2(x)], 1)

    with pytest.raises(AssertionError):
        PytorchGraphModelSpace.from_model(Net())
    with pytest.raises(ValueError):
        SimplifiedModelSpace.from_model(Net())


def test_discrete_in_evaluator(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix=None):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv2d(3, 5, kernel_size=Categorical([3, 5], label='ks'))

        def forward(self, x):
            return self.conv(x)

    def foo():
        pass

    evaluator = FunctionalEvaluator(foo, t=1, x=Categorical([1, 2], label='x'), y=Categorical([3, 4], label='y'))
    model = space_format.from_model(Net(), evaluator=evaluator)
    assert len(model.simplify()) == 3

    model1 = model.freeze({'ks': 3, 'x': 1, 'y': 3})
    assert model1.evaluator.get().arguments == {'t': 1, 'x': 1, 'y': 3}
    model2 = model.freeze({'ks': 5, 'x': 2, 'y': 4})
    assert model2.evaluator.get().arguments == {'t': 1, 'x': 2, 'y': 4}
    assert model2.executable_model()(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 5, 1, 1])


def test_model_evaluator_conflict_label(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv2d(3, 5, kernel_size=Categorical([3, 5], label='abc'))

        def forward(self, x):
            return self.conv(x)

    def foo():
        pass

    with label_scope('model'):
        evaluator = FunctionalEvaluator(foo, t=1, x=Categorical([3, 5], label='abc'))
    model = space_format.from_model(Net(), evaluator=evaluator)

    assert len(model.simplify()) == 1
    model1 = model.freeze({'model/abc': 5})
    assert model1.evaluator.get().arguments == {'t': 1, 'x': 5}
    assert model1.executable_model()(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 5, 1, 1])


def test_add_mutable(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.aux = self.add_mutable(Categorical([True, False]))
            self.head = ensure_frozen(self.aux)

        def forward(self, x):
            if self.head:
                return torch.ones_like(x)
            else:
                return torch.zeros_like(x)

        def freeze(self, sample):
            new_model = copy.deepcopy(self)
            new_model.head = self.aux.freeze(sample)
            return new_model

    if issubclass(space_format, GraphModelSpace):
        with pytest.raises(RuntimeError, match='Arbitrary'):
            space_format.from_model(Net())
    else:
        model = space_format.from_model(Net())
        assert len(model.simplify()) == 1
        label = list(model.simplify())[0]
        model1 = model.freeze({label: True})
        model2 = model.freeze({label: False})
        assert (model1.executable_model()(torch.randn(10)) == 1).all()
        assert (model2.executable_model()(torch.randn(10)) == 0).all()


def test_mutable_in_nn_parameter(space_format: Type[ExecutableModelSpace], nn: ModuleType):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.aux = torch.nn.Parameter(
                torch.zeros(1, ensure_frozen(Categorical([64, 128, 256], label='a')), 3, 3)
            )

        def forward(self):
            return self.aux

        def freeze(self, sample):
            with model_context(sample):
                return self.__class__()

    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.choice = self.add_mutable(Categorical([64, 128, 256], label='a'))
            self.inner = Inner()
            self.dim = ensure_frozen(self.choice)

        def forward(self):
            param = self.inner()
            assert param.size(1) == self.dim
            return param

        def freeze(self, sample):
            with model_context(sample):
                return self.__class__()

    if issubclass(space_format, GraphModelSpace):
        with pytest.raises(RuntimeError, match='Arbitrary'):
            space_format.from_model(Net())
    else:
        model = space_format.from_model(Net())
        assert len(model.simplify()) == 1
        label = list(model.simplify())[0]
        model1 = model.freeze({label: 64})
        model2 = model.freeze({label: 256})
        assert model1.executable_model()().size(1) == 64
        assert model2.executable_model()().size(1) == 256


def test_freeze_layerchoice():
    import torch.nn as nn

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([nn.Conv2d(3, i, kernel_size=1) for i in range(1, 11)], label='layer')

        def forward(self, x):
            return self.module(x)

    orig_model = Net()

    for i in range(10):
        model = orig_model.freeze({'layer': i})
        inp = torch.randn(1, 3, 3, 3)
        a = getattr(orig_model.module, str(i))(inp)
        b = model(inp)
        assert torch.allclose(a, b)


def test_freeze_layerchoice_nested():
    import torch.nn as nn

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.module = LayerChoice([
                LayerChoice([
                    nn.Conv2d(3, 3, kernel_size=1),
                    nn.Conv2d(3, 4, kernel_size=1),
                    nn.Conv2d(3, 5, kernel_size=1)
                ], label='b'),
                nn.Conv2d(3, 1, kernel_size=1)
            ], label='a')

        def forward(self, x):
            return self.module(x)

    orig_model = Net()
    input = torch.randn(1, 3, 5, 5)

    a = getattr(getattr(orig_model.module, '0'), '0')(input)
    b = orig_model.freeze({'a': 0, 'b': 0})(input)
    assert torch.allclose(a, b)

    a = getattr(getattr(orig_model.module, '0'), '1')(input)
    b = orig_model.freeze({'a': 0, 'b': 1})(input)
    assert torch.allclose(a, b)

    a = getattr(orig_model.module, '1')(input)
    b = orig_model.freeze({'a': 1})(input)
    assert torch.allclose(a, b)

    with pytest.raises(SampleValidationError):
        orig_model.freeze({'a': 0, 'b': 3})
    orig_model.freeze({'a': 1, 'b': 3})

    assert isinstance(orig_model.freeze({'a': 0, 'b': 1}).module, nn.Conv2d)
    assert isinstance(orig_model.freeze({'a': 1, 'b': 0}).module, nn.Conv2d)
