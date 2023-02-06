import sys

import pytest

import nni
from nni.mutable import *
from nni.mutable.mutable import _mutable_equal
from nni.nas.evaluator import FunctionalEvaluator, FrozenEvaluator
from nni.nas.space import *


@nni.trace
class MyModelSpace(BaseModelSpace):
    def __init__(self):
        self.a = Categorical([1, 2, 3], label='a')
        self.b = Categorical([4, 5, 6, 7], label='b')

        if current_model() is not None:
            self.a1 = ensure_frozen(self.a)
            self.b1 = ensure_frozen(self.b)

    def call(self, x):
        return x + self.a1 + self.b1

    def leaf_mutables(self, is_leaf):
        yield from [self.a, self.b]

    def check_contains(self, sample):
        return self.a.check_contains(sample) or self.b.check_contains(sample)

    def freeze(self, sample):
        with model_context(sample):
            return MyModelSpace()

def foo(model, a):
    return model.call(a)


def test_keep_model_space():
    model_space = MyModelSpace()
    evaluator = FunctionalEvaluator(foo, a=Categorical([0, 1], label='c'))
    exec_model = RawFormatModelSpace.from_model(model_space, evaluator)
    assert repr(exec_model) == str(exec_model)
    assert exec_model.sample is None
    assert exec_model.status == ModelStatus.Initialized
    assert exec_model.metric is None
    with pytest.raises(NotImplementedError):
        exec_model._dump()
    assert _mutable_equal(exec_model.simplify(), {
        'a': Categorical([1, 2, 3], label='a'),
        'b': Categorical([4, 5, 6, 7], label='b'),
        'c': Categorical([0, 1], label='c')
    })

    assert exec_model.contains({'a': 1, 'b': 4, 'c': -1}) is False
    assert exec_model.contains({'a': 0, 'b': 4, 'c': 0}) is False

    assert repr(exec_model).startswith('RawFormatModelSpace(model_space=MyModelSpace(), evaluator=FunctionalEvaluator(<function foo at ')
    if sys.platform == 'linux':
        # Otherwise trace will make the repr different
        assert repr(exec_model).endswith(">, arguments={'a': Categorical([0, 1], label='c')})), status=ModelStatus.Initialized)")
    frozen_model = exec_model.freeze({'a': 2, 'b': 6, 'c': 1})
    assert frozen_model.status == ModelStatus.Frozen
    assert frozen_model.sample == {'a': 2, 'b': 6, 'c': 1}
    assert frozen_model.evaluator.evaluate(frozen_model.executable_model()) == 9
    frozen_model.metrics.add_intermediate(1)
    frozen_model.metrics.final = 2
    assert repr(frozen_model).endswith(', metrics=Metrics(intermediates=<array of length 1>, final=2.0), status=ModelStatus.Frozen)')
    assert str(frozen_model) == "RawFormatModelSpace({'a': 2, 'b': 6, 'c': 1}, Metrics(intermediates=<array of length 1>, final=2.0), 'frozen')"

    with pytest.raises(RuntimeError, match='not initialized'):
        frozen_model.freeze({'a': 1, 'b': 5, 'c': 0})


def test_simplified_model_space():
    model_space = MyModelSpace()
    evaluator = FunctionalEvaluator(foo, a=Categorical([0, 1], label='c'))
    exec_model = SimplifiedModelSpace.from_model(model_space, evaluator)
    assert repr(exec_model) == str(exec_model)
    assert exec_model.status == ModelStatus.Initialized
    assert exec_model.metric is None
    expected_dump_result = {
        'status': ModelStatus.Initialized,
        'model_symbol': getattr(MyModelSpace, '__wrapped__', MyModelSpace),
        'model_args': [],
        'model_kwargs': {},
        'evaluator': FunctionalEvaluator(
            function=foo,
            a=Categorical([0, 1], label='c')
        ),
        'mutables': MutableDict({
            'a': Categorical([1, 2, 3], label='a'),
            'b': Categorical([4, 5, 6, 7], label='b')
        })
    }
    assert exec_model._dump() == expected_dump_result
    assert SimplifiedModelSpace._load(**expected_dump_result)._dump() == expected_dump_result

    assert _mutable_equal(exec_model.simplify(), {
        'a': Categorical([1, 2, 3], label='a'),
        'b': Categorical([4, 5, 6, 7], label='b'),
        'c': Categorical([0, 1], label='c')
    })
    assert exec_model.contains({'a': 1, 'b': 4, 'c': -1}) is False
    assert exec_model.contains({'a': 0, 'b': 4, 'c': 0}) is False

    assert repr(exec_model).endswith('status=ModelStatus.Initialized)')
    frozen_model = exec_model.freeze({'a': 2, 'b': 6, 'c': 1})
    assert frozen_model.sample == {'a': 2, 'b': 6, 'c': 1}
    assert frozen_model.status == ModelStatus.Frozen
    assert frozen_model.evaluator.evaluate(frozen_model.executable_model()) == 9
    frozen_model.metrics.add_intermediate(1)
    frozen_model.metrics.final = 2
    assert str(frozen_model) == "SimplifiedModelSpace({'a': 2, 'b': 6, 'c': 1}, Metrics(intermediates=<array of length 1>, final=2.0), 'frozen')"

    expected_dump_result = {
        'status': ModelStatus.Frozen,
        'model_symbol': getattr(MyModelSpace, '__wrapped__', MyModelSpace),
        'model_args': [],
        'model_kwargs': {},
        'sample': {'a': 2, 'b': 6, 'c': 1},
        'metrics': frozen_model.metrics
    }
    dump_result = frozen_model._dump()
    assert isinstance(dump_result['evaluator'], FrozenEvaluator) and dump_result['evaluator'].trace_kwargs['a'] == 1
    assert SimplifiedModelSpace._load(**dump_result)._dump() == dump_result
    dump_result.pop('evaluator')
    assert dump_result == expected_dump_result
    assert repr(frozen_model).endswith(', metrics=Metrics(intermediates=<array of length 1>, final=2.0), status=ModelStatus.Frozen)')

    with pytest.raises(RuntimeError, match='not initialized'):
        frozen_model.freeze({'a': 1, 'b': 5, 'c': 0})


def test_model_status():
    status = ModelStatus.Initialized
    assert not status.frozen()
    assert not status.completed()
    status = ModelStatus.Frozen
    assert status.frozen()
    assert not status.completed()

    assert status == 'frozen'
    assert status != 'initialized'

    status = ModelStatus.Trained
    assert status.frozen() and status.completed()
    status = ModelStatus.Interrupted
    assert status.frozen() and status.completed()
    status = ModelStatus.Failed
    assert status.frozen() and status.completed()
