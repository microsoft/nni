import copy

import pytest

import torch
import nni
from torch import nn
from nni.common.serializer import is_traceable
from nni.mutable import Categorical, ExpressionConstraint, ensure_frozen, frozen_context, label_scope, auto_label, SampleMissingError
from nni.mutable.mutable import _mutable_equal
from nni.nas.nn.pytorch import ModelSpace, MutableModule, ParametrizedModule
from nni.nas.space import model_context


def test_label_scope():
    from nni.nas.nn.pytorch.base import strict_label_scope

    with strict_label_scope('_unused_'):
        with pytest.raises(ValueError, match='Label'):
            Categorical([1, 2, 3])
        assert Categorical([1, 2, 3], label='x').label == 'x'
        with label_scope('hello'):
            assert auto_label() == 'hello/1'
            assert Categorical([1, 2, 3], label='x').label == 'hello/x'


def test_ensure_frozen():
    discrete = Categorical([1, 2, 3])
    with pytest.raises(RuntimeError, match='No frozen context'):
        ensure_frozen(discrete)

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.module = Categorical([1, 2, 3], label='x')
            self.add_mutable(self.module)
            self.dry_run_value = ensure_frozen(self.module)

    model = Net()
    assert model.dry_run_value == 1

    with pytest.raises(NotImplementedError, match='nni.nas.space.model_context'):
        model.freeze({'x': 2})


def test_ensure_frozen_freeze():
    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            self.module = Categorical([1, 2, 3], label='x')
            self.add_mutable(self.module)
            self.dry_run_value = ensure_frozen(self.module)

        def freeze(self, sample):
            with model_context(sample):
                return self.__class__()

    model = Net()
    assert model.dry_run_value == 1
    assert model.contains({'x': 1})
    assert not model.contains({})
    assert not model.contains({'x': 4})
    model1 = model.freeze({'x': 2})
    assert model1.dry_run_value == 2
    model2 = model.freeze({'x': 3})
    assert model2.dry_run_value == 3


def test_ensure_frozen_freeze_fail(caplog):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = Categorical([1, 2, 3], label='x')
            self.add_mutable(self.module)
            self.dry_run_value = ensure_frozen(self.module)

        def freeze(self, sample):
            with model_context(sample):
                return self.__class__()

    model = Net()
    assert model.dry_run_value == 1
    with pytest.raises(SampleMissingError):
        model.freeze({'x': 2})
    assert 'add_mutable()' in caplog.text


def test_ensure_frozen_no_register(caplog):
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = Categorical([1, 2, 3])
            self.dry_run_value = ensure_frozen(self.module)

    with pytest.raises(SampleMissingError):
        Net()
    assert 'Failed to freeze mutable' in caplog.text


def test_ensure_frozen_in_forward():
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.module = Categorical([1, 2, 3])
            self.add_mutable(self.module)

        def forward(self):
            return ensure_frozen(self.module)

    model = Net()
    with pytest.raises(RuntimeError, match='in forward'):
        model()


def test_ensure_frozen_constraint_first():
    class Net(ModelSpace, label_prefix='model'):
        def __init__(self, s: int = 4):
            super().__init__()
            discrete1 = Categorical([1, 2, 3])
            discrete2 = Categorical([1, 2, 3])
            self.add_mutable(ExpressionConstraint(discrete1 + discrete2 == s))
            self.add_mutable(discrete1)
            self.add_mutable(discrete2)
            self.a = ensure_frozen(discrete1)
            self.b = ensure_frozen(discrete2)

    model = Net()

    # assert 'Using the default' in caplog.text
    assert model.a + model.b == 4

    model = Net(5)
    assert model.a + model.b == 5

    with pytest.raises(
        ValueError,
        match=r'Cannot find a valid',
    ):
        model = Net(10)


@pytest.mark.parametrize('label_prefix', [None, 'model'])
def test_ensure_frozen_consistency(label_prefix):
    class Submodule(MutableModule):
        def __init__(self):
            super().__init__()
            discrete2 = Categorical([1, 2, 3], default=3, label='a')
            self.add_mutable(discrete2)
            self.b = ensure_frozen(discrete2)

        def forward(self, x):
            return self.conv(x)

    class Net(ModelSpace):
        def __init__(self):
            super().__init__()
            discrete1 = Categorical([1, 2, 3], default=2, label='a')
            self.add_mutable(discrete1)
            self.a = ensure_frozen(discrete1)
            self.sub = Submodule()

    with pytest.raises(ValueError, match='default value of'):
        Net()


def test_model_space_args():
    class ModelSpace1(ModelSpace):
        def __init__(self, a, b=1):
            super().__init__()
            self.a = a
            self.b = b

    model = ModelSpace1(2, 3)
    assert model.args['a'] == 2
    assert model.args['b'] == 3

    model = ModelSpace1(2)
    assert model.args['a'] == 2
    assert model.args['b'] == 1
    assert 'b' not in model.trace_kwargs

    class ModelSpace2(ModelSpace):
        def __init__(self, *args):
            super().__init__()
            self.a = args

    model = ModelSpace2(1, 2, 3)
    with pytest.raises(RuntimeError, match='args is not available'):
        model.args

    assert is_traceable(model)


def test_model_space_label():

    class MyModelSpace(ModelSpace, label_prefix='backbone'):
        def __init__(self):
            super().__init__()

            self.choice = self.add_mutable(nni.choice('depth', [2, 3, 4]))

    model = MyModelSpace()
    assert model.choice.label == 'backbone/depth'



def test_model_space_inherit():
    class ModelSpace1(ModelSpace):
        def __init__(self, a, b=1):
            super().__init__()
            self.d = self.add_mutable(Categorical([4, 5, 6], label='d'))

    class ModelSpace2(ModelSpace1):
        def __init__(self, c=2):
            super().__init__(c, b=3)
            self.c = self.add_mutable(Categorical([1, 2, 3], label='c'))

    model = ModelSpace2(5)
    assert model.trace_kwargs == {'c': 5}
    assert model.c.label == 'c'
    assert model.d.label == 'd'
    assert model.trace_symbol == ModelSpace2


def test_model_space_inherit_label_prefix():
    class ModelSpace1(ModelSpace):
        def __init__(self, a, b=1):
            super().__init__()
            self.d = self.add_mutable(Categorical([4, 5, 6], label='d'))

    class ModelSpace2(ModelSpace1, label_prefix='model2'):
        def __init__(self, c=2):
            super().__init__(c, b=3)
            self.c = self.add_mutable(Categorical([1, 2, 3], label='c'))

    model = ModelSpace2(5)
    assert model.trace_kwargs == {'c': 5}
    assert model.c.label == 'model2/c'
    assert model.d.label == 'model2/d'
    assert model.trace_symbol == ModelSpace2


def test_model_space_combination_label_prefix():
    class ModelSpace1(ModelSpace, label_prefix='model1'):
        def __init__(self, a, b=1):
            super().__init__()
            self.d = self.add_mutable(Categorical([4, 5, 6], label='d'))

    class ModelSpace2(ModelSpace, label_prefix='model2'):
        def __init__(self, c=2):
            super().__init__()
            self.s = ModelSpace1(1, 2)
            self.c = self.add_mutable(Categorical([1, 2, 3], label='c'))

    model = ModelSpace2(5)
    assert model.trace_kwargs == {'c': 5}
    assert model.c.label == 'model2/c'
    assert model.s.d.label == 'model2/model1/d'
    assert model.trace_symbol == ModelSpace2
    assert model.s.trace_symbol == ModelSpace1


def test_model_space_no_label_prefix(caplog):
    class ModelSpace1(ModelSpace, label_prefix=''):
        pass

    with pytest.raises(ValueError, match='be empty'):
        ModelSpace1()

    class ModelSpace2(ModelSpace):
        def __init__(self):
            super().__init__()
            self.a = self.add_mutable(Categorical([1, 2, 3], label='a'))
            assert self.a.label == 'a'
            with pytest.raises(ValueError, match='must be specified'):
                self.b = self.add_mutable(Categorical([1, 2, 3]))

    ModelSpace2()

    class ModelSpace3(ModelSpace):
        def __init__(self):
            super().__init__()
            self.a = self.add_mutable(Categorical([1, 2, 3], label='a'))
            assert self.a.label == 'model4/a'
            with pytest.raises(ValueError, match='must be specified'):
                self.b = self.add_mutable(Categorical([1, 2, 3]))

    class ModelSpace4(ModelSpace, label_prefix='model4'):
        def __init__(self):
            super().__init__()
            ModelSpace3()
            self.a = self.add_mutable(Categorical([1, 2, 3], label='a'))
            assert self.a.label == 'model4/a'
            self.b = self.add_mutable(Categorical([1, 2, 3]))
            assert self.b.label == 'model4/1'

    class ModelSpace5(ModelSpace):
        def __init__(self):
            super().__init__()
            ModelSpace4()

    ModelSpace5()



def test_import_nas_nn_as_nn():
    import torch
    import nni.nas.nn.pytorch.layers as nn

    dummy = torch.zeros(1, 16, 32, 24)
    nn.init.uniform_(dummy)

    nn.Conv2d(1, 3, 1)
    nn.Parameter(torch.zeros(1, 3, 24, 24))

    import nni.nas.nn.pytorch as nn2

    nn2.MutableConv2d(1, 3, 1)
    with pytest.raises(AttributeError):
        nn2.Conv2d(1, 3, 1)


def test_label():
    from nni.nas.nn.pytorch import LayerChoice

    class Model(ModelSpace, label_prefix='model'):
        def __init__(self, in_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 10, 3)
            self.conv2 = LayerChoice([
                nn.Conv2d(10, 10, 3),
                nn.MaxPool2d(3)
            ])
            self.conv3 = LayerChoice([
                nn.Identity(),
                nn.Conv2d(10, 10, 1)
            ])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.avgpool(x).view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = Model(3)
    assert model.trace_symbol == Model
    assert model.trace_kwargs == {'in_channels': 3}
    assert model.conv2.label == 'model/1'
    assert model.conv3.label == 'model/2'
    assert model(torch.randn(1, 3, 5, 5)).size() == torch.Size([1, 1])

    model = Model(4)
    assert model.trace_symbol == Model
    assert model.conv2.label == 'model/1'  # not changed


def test_label_hierarchy():
    from nni.nas.nn.pytorch import LayerChoice

    class ModelInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.net1 = LayerChoice([
                nn.Linear(10, 10),
                nn.Linear(10, 10, bias=False)
            ])
            self.net2 = LayerChoice([
                nn.Linear(10, 10),
                nn.Linear(10, 10, bias=False)
            ])

        def forward(self, x):
            x = self.net1(x)
            x = self.net2(x)
            return x

    class ModelNested(ModelSpace, label_prefix='model'):
        def __init__(self):
            super().__init__()
            self.fc1 = ModelInner()
            self.fc2 = LayerChoice([
                nn.Linear(10, 10),
                nn.Linear(10, 10, bias=False)
            ])
            self.fc3 = ModelInner()

        def forward(self, x):
            return self.fc3(self.fc2(self.fc1(x)))

    model = ModelNested()
    assert model.fc1.net1.label == 'model/1'
    assert model.fc1.net2.label == 'model/2'
    assert model.fc2.label == 'model/3'
    assert model.fc3.net1.label == 'model/4'
    assert model.fc3.net2.label == 'model/5'


def test_deepcopy():
    class MyModule(MutableModule):
        def __init__(self, a, b):
            super().__init__()
            self.a = nn.Linear(3, a)
            self.b = nn.Linear(3, b)
            self.mutable = self.add_mutable(Categorical([0, 1]))

            self.flag = ensure_frozen(self.mutable)

        def forward(self, x):
            if self.flag:
                return self.a(x)
            else:
                return self.b(x)

    with frozen_context():
        net = MyModule(5, 7)
    net2 = copy.deepcopy(net)
    assert net2.a.out_features == 5
    assert net2.b.out_features == 7
    assert net2.flag == net.flag


def test_mutable_descendants():
    class A(MutableModule):
        pass

    class B(MutableModule):
        def check_contains(self, sample):
            if 'test' in sample:
                return SampleMissingError('test')

    class C(MutableModule):
        def __init__(self):
            super().__init__()
            self.b = B()

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = C()

    class E(MutableModule):
        def __init__(self):
            super().__init__()
            self.a = A()
            self.d = D()

    e = E()
    descendants = list(e.mutable_descendants())
    assert len(descendants) == 2
    assert [_[0] for _ in e.named_mutable_descendants()] == ['a', 'd.c']
    assert isinstance(descendants[0], A)
    assert isinstance(descendants[1], C)

    assert repr(e) == """E(
  (a): A()
  (d): D(
    (c): C(
      (b): B()
    )
  )
)"""
    frozen_e = e.freeze({})
    assert e is not frozen_e
    assert repr(e) == repr(frozen_e)

    e.validate({})
    with pytest.raises(SampleMissingError, match='d.c'):
        e.validate({'test': 1})


def test_parametrized_module():
    class MyModule(ParametrizedModule):
        def __init__(self, x):
            super().__init__()
            self.t = x

    with frozen_context():
        module = MyModule(nni.choice('choice1', [1, 2, 3]))
    assert module.t == 1
    assert module.args['x'].label == 'choice1'

    assert repr(module) == "MyModule(x=Categorical([1, 2, 3], label='choice1'))"
    assert str(module) == repr(module)

    class ParametrizedConv2d(ParametrizedModule, nn.Conv2d, wraps=nn.Conv2d):
        pass

    with pytest.raises(RuntimeError):
        ParametrizedConv2d(3, nni.choice('out', [8, 16]), 1)

    with frozen_context():
        conv = ParametrizedConv2d(3, nni.choice('out', [8, 16]), 1)

    assert repr(conv) == "ParametrizedConv2d(in_channels=3, out_channels=Categorical([8, 16], label='out'), kernel_size=1)"
    assert conv.out_channels == 8
    assert conv.args['padding'] == 0
    assert _mutable_equal(conv.args['out_channels'], nni.choice('out', [8, 16]))
    assert conv.contains({'out': 16})
    assert len(conv.simplify()) == 1
    assert type(conv.freeze({'out': 16})) is nn.Conv2d
    assert conv.freeze({'out': 16}).out_channels == 16


def test_parametrized_module_inheritance():
    class Module1(ParametrizedModule):
        def __init__(self, x):
            super().__init__()
            self.t = x

    class Module2(Module1):
        def __init__(self, y):
            super().__init__(y)

    choice = nni.choice('choice1', [1, 2, 3])
    with frozen_context():
        module = Module2(choice)
    assert _mutable_equal(module.args, {'y': choice})
    assert module.freeze({'choice1': 2}).t == 2


def test_nested_parametrized_module():
    class MyModule(ParametrizedModule):
        def __init__(self, x):
            if x == 0:
                self.mutable = self.add_mutable(nni.choice('a', [1, 2, 3]))
            else:
                self.mutable = self.add_mutable(nni.choice('b', [4, 5, 6]))
            self.flag = ensure_frozen(self.mutable)

    with frozen_context():
        module = MyModule(nni.choice('x', [0, 1]))
    module1 = module.freeze({'x': 0, 'a': 1})
    assert module1.flag == 1
    module2 = module.freeze({'x': 1, 'b': 5})
    assert module2.mutable.label == 'b'
    assert module2.flag == 5
    with pytest.raises(SampleMissingError):
        module.freeze({'x': 1, 'a': 2})


def test_empty_parameterized_module():
    class MutableConv(ParametrizedModule):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv2 = nn.Conv2d(3, 5, kernel_size=1)

        def forward(self, x: torch.Tensor, index: int):
            if index == 0:
                return self.conv1(x)
            else:
                return self.conv2(x)

    class MyModelSpace(ModelSpace):
        def __init__(self):
            super().__init__()
            self.conv = MutableConv()
            self.index = ensure_frozen(self.add_mutable(nni.choice('x', [0, 1])))

        def freeze(self, sample):
            with model_context(sample):
                return self.__class__()

        def forward(self, x: torch.Tensor):
            return self.conv(x, self.index)

    space = MyModelSpace()
    model = space.freeze({'x': 0})
    assert model(torch.randn(1, 3, 5, 5)).shape == (1, 3, 5, 5)
    model = space.freeze({'x': 1})
    assert model(torch.randn(1, 3, 5, 5)).shape == (1, 5, 5, 5)
