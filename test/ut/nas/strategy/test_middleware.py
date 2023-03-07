import random
from collections import defaultdict, Counter

import pytest

from nni.mutable import SampleValidationError
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.space import ModelStatus
from nni.nas.strategy import Random, RegularizedEvolution, PolicyBasedRL, GridSearch, Strategy
from nni.nas.strategy.base import StrategyStatus
from nni.nas.strategy.middleware import Filter, Chain, Deduplication, FailureHandler, MultipleEvaluation

from .test_sanity import named_model_space, engine, naive_policy


class MockStrategy(Strategy):
    def __init__(self, model_generator=None, auto_check=True):
        super().__init__()
        self.model_generator = model_generator
        self.auto_check = auto_check

        self._check_intermediate_events = defaultdict(list)
        self._check_final_events = defaultdict(list)
        self._check_status_events = defaultdict(list)

    def _initialize(self, model_space, engine):
        engine.register_model_event_callback('intermediate_metric', self._intermediate_metric)
        engine.register_model_event_callback('final_metric', self._final_metric)
        engine.register_model_event_callback('training_end', self._training_end)
        return model_space

    def _cleanup(self):
        self.engine.unregister_model_event_callback('intermediate_metric', self._intermediate_metric)
        self.engine.unregister_model_event_callback('final_metric', self._final_metric)
        self.engine.unregister_model_event_callback('training_end', self._training_end)

    def _run(self):
        while self.engine.budget_available():
            if self.engine.idle_worker_available():
                if self.model_generator is None:
                    try:
                        model = self.model_space.random()
                    except SampleValidationError:
                        continue
                    self.engine.submit_models(model)
                else:
                    self.engine.submit_models(next(self.model_generator))

        self.engine.wait_models()
        self._models = list(self.engine.list_models())

        if self.auto_check:
            for model in self._models:
                assert model.metrics.intermediates == self._check_intermediate_events[model]
                if model.metrics.final is None:
                    assert [] == self._check_final_events[model]
                else:
                    assert [model.metrics.final] == self._check_final_events[model]
                assert [model.status] == self._check_status_events[model]

    def state_dict(self):
        return {'hello': 'world'}

    def load_state_dict(self, state_dict: dict) -> None:
        assert state_dict['hello'] == 'world'

    def _intermediate_metric(self, event):
        self._check_intermediate_events[event.model].append(event.metric)

    def _final_metric(self, event):
        self._check_final_events[event.model].append(event.metric)

    def _training_end(self, event):
        self._check_status_events[event.model].append(event.status)


@pytest.mark.parametrize('strategy', ['grid', 'random', 'evolution', 'rl'])
@pytest.mark.parametrize('metric_for_invalid', [-1., None])
@pytest.mark.parametrize('retain_history', [False, True])
def test_filter(named_model_space, engine, strategy, metric_for_invalid, retain_history):
    name, model_space = named_model_space

    if strategy == 'grid':
        strategy = GridSearch(dedup=False)
    elif strategy == 'random':
        strategy = Random(dedup=False)
    elif strategy == 'evolution':
        strategy = RegularizedEvolution(dedup=False)
    elif strategy == 'rl':
        strategy = PolicyBasedRL(policy_fn=naive_policy)

    def filter_fn(model):
        return model.sample['a'] <= 2

    strategy = Chain(
        Random(dedup=False),
        Filter(filter_fn, metric_for_invalid=metric_for_invalid, retain_history=retain_history)
    )

    if metric_for_invalid is None:
        assert repr(strategy) == f"""Chain(
  Random(dedup=False),
  Filter(filter_fn={filter_fn})
)"""

    strategy(model_space, engine)
    assert strategy._status == StrategyStatus.SUCCEEDED
    assert all(not v for v in engine._callbacks.values())

    if metric_for_invalid is not None:
        any_invalid = False
        for model in strategy.list_models(sort=False):
            if model.sample['a'] > 2:
                assert model.metric == -1
                assert model.status == ModelStatus.Invalid
                any_invalid = True
        if retain_history:
            assert any_invalid
    else:
        assert all(model.sample['a'] <= 2 for model in engine.list_models())


def test_filter_patience(named_model_space, engine):
    name, model_space = named_model_space

    def gen():
        while True:
            try:
                model = model_space.random()
                if model.sample['a'] <= 2:
                    continue
                yield model
            except SampleValidationError:
                pass

    def filter_fn(model):
        return model.sample['a'] <= 2

    strategy = Chain(
        MockStrategy(gen()),
        Filter(filter_fn, metric_for_invalid=0.)
    )

    strategy(model_space, engine)
    assert strategy[1]._filter_count_patience >= strategy[1].patience
    assert len(engine.list_models()) == 0


def test_deduplication(named_model_space, engine):
    name, model_space = named_model_space

    strategy = Chain(
        MockStrategy(),
        Deduplication('invalid', retain_history=True)
    )
    strategy(model_space, engine)
    assert all(not v for v in engine._callbacks.values())

    models = list(strategy.list_models(sort=False))
    assert any(model.status == ModelStatus.Invalid for model in models)
    assert not all(model.status == ModelStatus.Invalid for model in models)

    for model in models:
        assert len([m for m in models if m.sample == model.sample]) >= 2


def test_deduplication_patience(named_model_space, engine):
    name, model_space = named_model_space

    def gen():
        same_model = None
        while True:
            try:
                if same_model is None:
                    same_model = model_space.random()
                yield same_model
            except SampleValidationError:
                pass

    strategy = Chain(
        MockStrategy(gen(), auto_check=False),
        Deduplication('invalid', retain_history=True)
    )

    strategy(model_space, engine)
    assert strategy[1]._dup_count_patience >= strategy[1].patience
    assert len(engine.list_models()) == 1


def test_deduplication_replay(named_model_space, engine):
    name, model_space = named_model_space

    strategy = Chain(
        MockStrategy(),
        Deduplication('replay', retain_history=True)
    )
    strategy(model_space, engine)
    assert len(engine.list_models()) < len(list(strategy.list_models()))

    for model in engine.list_models():
        for m in strategy.list_models():
            if model.sample == m.sample:
                assert model.metrics.intermediates == m.metrics.intermediates
                assert model.metrics.final == m.metrics.final
                assert model.status == m.status
                assert m.status != ModelStatus.Invalid


def test_failure_handler(named_model_space):
    name, model_space = named_model_space
    engine = SequentialExecutionEngine(max_model_count=30, continue_on_failure=True)

    def wrap(fn):
        def wrapper(*args, **kwargs):
            fn(*args, **kwargs)
            if random.random() < 0.5:
                raise RuntimeError('failed')
        return wrapper

    model_space.evaluator.function = wrap(model_space.evaluator.function)

    strategy = Chain(
        MockStrategy(auto_check=False),
        FailureHandler(metric=-1.)
    )
    strategy(model_space, engine)
    assert len(list(engine.list_models())) == len(list(strategy.list_models(sort=False)))

    any_failed = False
    for model in strategy.list_models(sort=False):
        if model.status == ModelStatus.Failed:
            assert model.metric == -1
            any_failed = True

        assert model.metrics.intermediates == strategy[0]._check_intermediate_events[model]
        assert model.metrics.final == strategy[0]._check_final_events[model][-1]
        assert [model.status] == strategy[0]._check_status_events[model]

    assert any_failed


def test_failure_handler_rerun(named_model_space):
    name, model_space = named_model_space
    engine = SequentialExecutionEngine(max_model_count=30, continue_on_failure=True)

    def wrap(fn):
        def wrapper(*args, **kwargs):
            fn(*args, **kwargs)
            if random.random() < 0.5:
                raise RuntimeError('failed')
        return wrapper

    model_space.evaluator.function = wrap(model_space.evaluator.function)

    mock_strategy = MockStrategy(auto_check=False)
    strategy = Chain(
        mock_strategy,
        FailureHandler(retry_patience=100, retain_history=True),
    )
    strategy(model_space, engine)
    assert len(list(engine.list_models())) > len(list(strategy.list_models(sort=False)))

    any_failed = False
    failure_status_count = 0
    for model in strategy.list_models(sort=False):
        if len(mock_strategy._check_final_events[model]) == 1:
            assert model.metrics.intermediates == mock_strategy._check_intermediate_events[model]
            assert [model.metrics.final] == mock_strategy._check_final_events[model]
            assert [model.status] == mock_strategy._check_status_events[model]
        else:
            retried = len(mock_strategy._check_final_events[model])
            any_failed = True
            assert model.metrics.intermediates * retried == mock_strategy._check_intermediate_events[model]
            assert [model.metrics.final] * retried == mock_strategy._check_final_events[model]
            assert [model.status] == mock_strategy._check_status_events[model]  # only one status
            if model.status != ModelStatus.Trained:
                failure_status_count += 1

    assert any_failed
    assert failure_status_count <= 1


@pytest.mark.parametrize('failure_types', [(ModelStatus.Invalid, ), (ModelStatus.Failed, )])
def test_failure_handler_filter(named_model_space, failure_types):
    name, model_space = named_model_space
    engine = SequentialExecutionEngine(max_model_count=30, continue_on_failure=True)

    _appeared = []

    def filter_fn(model):
        if model not in _appeared:
            _appeared.append(model)
            return model.sample['a'] <= 2
        if random.random() < 0.5:
            return model.sample['a'] <= 2
        return True

    mock_strategy = MockStrategy(auto_check=False)
    strategy = Chain(
        mock_strategy,
        FailureHandler(retry_patience=1, failure_types=failure_types),
        Filter(filter_fn)
    )
    strategy(model_space, engine)

    status_counter = Counter()
    for model in strategy.list_models(sort=False):
        if model.sample['a'] > 2:
            status_counter[model.status] += 1

    if failure_types[0] == ModelStatus.Failed:
        assert set(status_counter) == {ModelStatus.Invalid}
    else:
        assert set(status_counter) == {ModelStatus.Invalid, ModelStatus.Trained}


def test_multiple_evaluation(named_model_space, engine):
    name, model_space = named_model_space
    strategy = Chain(
        MockStrategy(),
        MultipleEvaluation(3)
    )

    strategy(model_space, engine)
    assert len(list(strategy.list_models(sort=False))) * 3 == len(list(engine.list_models()))


def test_multiple_evaluation_messy(named_model_space):
    name, model_space = named_model_space
    mock_strategy = MockStrategy(auto_check=False)
    strategy = Chain(
        mock_strategy,
        MultipleEvaluation(3)
    )
    engine = SequentialExecutionEngine(max_model_count=30, continue_on_failure=True)

    import nni

    def wrap(fn):
        def wrapper(*args, **kwargs):
            if random.random() < 0.5:
                nni.report_intermediate_result(0.)
            fn(*args, **kwargs)
            if random.random() < 0.3:
                nni.report_final_result(1.)
            if random.random() < 0.3:
                raise RuntimeError('failed')
        return wrapper

    model_space.evaluator.function = wrap(model_space.evaluator.function)

    strategy(model_space, engine)
    assert len(list(strategy.list_models(sort=False))) * 3 == len(list(engine.list_models()))

    for model in strategy.list_models(sort=False):
        assert 1 <= len(model.metrics.intermediates) <= 2
        assert model.metrics.intermediates == mock_strategy._check_intermediate_events[model]
        assert model.metrics.final == mock_strategy._check_final_events[model][-1]
        assert [model.status] == mock_strategy._check_status_events[model]
        assert model.status in [ModelStatus.Trained, ModelStatus.Failed]


@pytest.mark.parametrize('middleware', ['filter', 'dedup', 'failure', 'multiple'])
def test_state_dict_sanity(middleware, named_model_space, engine):
    def middleware_fn():
        if middleware == 'filter':
            return Filter(lambda model: model.sample['a'] >= 2)
        elif middleware == 'dedup':
            return Deduplication('replay')
        elif middleware == 'failure':
            return FailureHandler(metric=0.)
        elif middleware == 'multiple':
            return MultipleEvaluation(4)

    strategy = Chain(
        MockStrategy(),
        middleware_fn()
    )

    _, model_space = named_model_space
    strategy(model_space, engine)

    state_dict = strategy.state_dict()
    strategy2 = Chain(
        MockStrategy(auto_check=False),
        middleware_fn()
    )
    strategy2.load_state_dict(state_dict)

    engine.max_model_count += 10
    strategy2(model_space, engine)
