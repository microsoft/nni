import re
import sys
import pytest

import nni
import nni.trial
import torch
import pytorch_lightning

from nni.mutable import Categorical, Numerical, SampleValidationError
from nni.mutable.mutable import _mutable_equal
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.evaluator.pytorch import Classification, Lightning, DataLoader, Trainer, LightningModule
from nni.nas.evaluator.evaluator import FrozenEvaluator
from torch import nn
from torch.utils.data import TensorDataset


@pytest.fixture(autouse=True)
def reset_cached_parameter():
    nni.trial._params = None
    nni.trial.overwrite_intermediate_seq(0)


@nni.trace
def _print_params(m, a, b):
    print(a, b)


@nni.trace
def custom_function(a, b):
    return a + b


@nni.trace
class custom_class:
    def __init__(self, a, b):
        assert isinstance(a, int) and isinstance(b, int)
        self.c = a + b

    def func(self):
        return self.c + 1

    def __repr__(self):
        return f'c {self.c}'


@pytest.mark.parametrize('trace_decorator', [False, True])
def test_functional_mutate(capsys, trace_decorator):
    if trace_decorator:
        # The evaluator works with or without trace decorator.
        cls = nni.trace(FunctionalEvaluator)
    else:
        cls = FunctionalEvaluator
    evaluator = cls(_print_params, a=Categorical([1, 2], label='x'), b=Categorical([3, 4], label='y'))
    assert _mutable_equal(evaluator.simplify(), {'x': Categorical([1, 2], label='x'), 'y': Categorical([3, 4], label='y')})
    _dump_result = {
        'a': Categorical([1, 2], label='x'),
        'b': Categorical([3, 4], label='y'),
        'function': _print_params,
    }
    assert evaluator.is_mutable()
    if not trace_decorator:
        assert re.match(r"FunctionalEvaluator\(<.*>, arguments={'a': Categorical.*, 'b': Categorical.*", repr(evaluator))
    assert _mutable_equal(evaluator._dump(), _dump_result)
    assert cls._load(**_dump_result) == evaluator

    evaluator = evaluator.freeze({'x': 1, 'y': 3})
    assert isinstance(evaluator, FrozenEvaluator)
    _dump_result = {
        'trace_symbol': FunctionalEvaluator,
        'trace_kwargs': {'function': _print_params, 'a': 1, 'b': 3},
        'trace_args': []
    }
    assert evaluator._dump() == _dump_result
    FrozenEvaluator._load(**_dump_result) == evaluator

    evaluator.evaluate(None)
    assert capsys.readouterr().out == '1 3\n'
    assert evaluator.get() is evaluator.get()
    assert not evaluator.get().is_mutable()


def test_functional_nested(capsys):
    evaluator = FunctionalEvaluator(
        _print_params,
        a=custom_function(1, Numerical(0, 1, label='x')),
        b=custom_class(Categorical([3, 4], label='y'), Categorical([5, 6], label='z'))
    )

    evaluator = evaluator.freeze({'x': 0.5, 'y': 3, 'z': 6})
    assert isinstance(evaluator, FrozenEvaluator)
    with pytest.raises(AttributeError):
        evaluator.trace_kwargs['b'].func()
    assert evaluator.get().arguments['a'] == 1.5
    assert evaluator.get().arguments['b'].func() == 10

    evaluator.evaluate(None)
    assert capsys.readouterr().out == '1.5 c 9\n'


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Lightning 1.0 required')
def test_classification_mutate():
    evaluator = Classification(
        criterion=nn.CrossEntropyLoss,
        learning_rate=Categorical([0.1, 0.01], label='lr'),
        train_dataloaders=DataLoader(TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,))),
                                     batch_size=Categorical([2, 4], label='bs')),
        val_dataloaders=DataLoader(TensorDataset(torch.randn(10, 3), torch.randint(0, 2, (10,))), batch_size=2),
        max_epochs=Categorical([1, 2], label='epoch'),
        num_classes=2,
    )
    with pytest.raises(RuntimeError, match='freeze()'):
        evaluator.evaluate(nn.Linear(3, 2))
    evaluator = evaluator.freeze({'lr': 0.1, 'bs': 2, 'epoch': 2})
    assert isinstance(evaluator, FrozenEvaluator)
    assert evaluator.trace_kwargs['learning_rate'] == 0.1
    assert evaluator.trace_kwargs['train_dataloaders'].trace_kwargs['batch_size'] == 2
    assert evaluator.trace_kwargs['max_epochs'] == 2
    assert not isinstance(evaluator.trace_kwargs['train_dataloaders'], DataLoader)
    assert isinstance(evaluator.trace_kwargs['val_dataloaders'], DataLoader)

    evaluator.evaluate(nn.Linear(3, 2))
    evaluator = evaluator.get()
    assert isinstance(evaluator, Lightning)
    assert evaluator.trainer.max_epochs == 2


def test_mutable_expression_in_evaluator():
    def foo():
        pass

    evaluator = FunctionalEvaluator(foo, t=1, x=2)
    assert len(evaluator.simplify()) == 0

    evaluator = FunctionalEvaluator(foo, t=1, x=Categorical([1, 2], label='x'), y=Categorical([3, 4], label='y'))
    assert len(evaluator.simplify()) == 2

    evaluator1 = evaluator.freeze({'x': 1, 'y': 3})
    evaluator2 = evaluator.freeze({'x': 2, 'y': 4})

    with pytest.raises(SampleValidationError):
        evaluator.freeze({'x': 3, 'y': 4})

    assert evaluator1.get().arguments == {'t': 1, 'x': 1, 'y': 3}
    assert evaluator2.get().arguments == {'t': 1, 'x': 2, 'y': 4}

    # share label
    evaluator = FunctionalEvaluator(foo, t=Categorical([1, 2], label='x'), x=Categorical([1, 2], label='x'))
    assert len(evaluator.simplify()) == 1

    # getitem
    choice = Categorical([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    evaluator = FunctionalEvaluator(foo, t=1, x=choice['a'], y=choice['b'])
    assert len(evaluator.simplify()) == 1
    for _ in range(10):
        evaluator1 = evaluator.random()
        assert (evaluator1.trace_kwargs['x'], evaluator1.trace_kwargs['y']) in [(1, 2), (3, 4)]


def test_evaluator_mutable_nested():
    @nni.trace
    class FooClass:
        def __init__(self, a):
            self.a = a

    obj = FooClass(Categorical([1, 2, 3], label='t'))

    def foo():
        pass

    evaluator = FunctionalEvaluator(foo, t=obj, v=Categorical([1, 2, 3], label='t') + Categorical([10, 20, 30]))
    assert len(evaluator.simplify()) == 2
    for _ in range(10):
        evaluator1 = evaluator.random()
        a, v = evaluator1.trace_kwargs['t'].trace_kwargs['a'], evaluator1.trace_kwargs['v']
        assert v % 10 == a
        assert a in [1, 2, 3]
        assert v // 10 in [1, 2, 3]


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Legacy PyTorch-lightning not supported')
def test_choice_in_lightning():
    @nni.trace
    class AnyModule(LightningModule):
        pass

    evaluator = Lightning(AnyModule(), Trainer(max_epochs=Categorical([1, 2, 3])))
    assert len(evaluator.simplify()) == 1
    label = list(evaluator.simplify())[0]
    evaluator1 = evaluator.freeze({label: 2})
    assert evaluator1.get().trainer.max_epochs in [1, 2, 3]

    for target, evaluator1 in enumerate(evaluator.grid(), start=1):
        assert evaluator1.get().trainer.max_epochs == target


@pytest.mark.skipif(pytorch_lightning.__version__ < '1.0', reason='Legacy PyTorch-lightning not supported')
def test_choice_in_classification():
    evaluator = Classification(criterion=nn.CrossEntropyLoss, num_classes=2)
    assert len(evaluator.simplify()) == 0
    evaluator.freeze({})


def test_mock_trial_api(caplog):
    from nni.nas.space import RawFormatModelSpace, BaseModelSpace

    class DummyModelSpace(BaseModelSpace):
        def check_contains(self, sample):
            return None

        def leaf_mutables(self, is_leaf):
            yield from ()

        def freeze(self, sample):
            return self

    def foo(model):
        import nni
        nni.report_intermediate_result(0.5)
        assert 'Intermediate metric: 0.5' in caplog.text
        nni.report_final_result(0.6)
        assert 'Final metric: 0.6' in caplog.text

    space = DummyModelSpace()
    space_cvt = RawFormatModelSpace.from_model(space)
    model = space_cvt.random()
    evaluator = FunctionalEvaluator(foo)

    with pytest.raises(TypeError, match='ExecutableModelSpace'):
        m = model.executable_model()
        with evaluator.mock_runtime(m):
            evaluator.evaluate(m)

    with evaluator.mock_runtime(model):
        import nni
        assert nni.get_current_parameter() == model
        evaluator.evaluate(model.executable_model())
        assert '[Mock] Final' in caplog.text

    assert nni.get_current_parameter() is None
