import time
import pytest

import nni

from nni.nas.experiment import NasExperiment
from nni.nas.execution import ExecutionEngine, TrainingServiceExecutionEngine, SequentialExecutionEngine
from nni.nas.execution.event import ModelEventType
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.space import BaseModelSpace, SimplifiedModelSpace, current_model, ModelStatus

@nni.trace
class MyModelSpace(BaseModelSpace):
    def __init__(self, a):
        if current_model() is not None:
            self.a = current_model()['a']
        else:
            self.a = nni.choice('a', a)

    def call(self, x):
        return x + self.a

    def leaf_mutables(self, is_leaf):
        yield self.a


def evaluate_fn(model, b):
    nni.report_intermediate_result(model.call(1))
    nni.report_intermediate_result(model.call(2))
    if model.a == 3:
        raise RuntimeError()
    nni.report_final_result(model.call(3) + b)


@pytest.fixture(params=['sequential', 'ts'])
def engine(request):
    if request.param == 'sequential':
        yield SequentialExecutionEngine(continue_on_failure=True)
    elif request.param == 'ts':
        nodejs = NasExperiment(None, None, None)
        nodejs._start_nni_manager(8080, True)
        yield TrainingServiceExecutionEngine(nodejs)
        nodejs._stop_nni_manager()


def test_engine(engine: ExecutionEngine):
    _callback_counter = 0
    _intermediates = []
    _finals = []
    _status = []
    _callback_disabled = False

    def callback(event):
        if _callback_disabled:
            return

        nonlocal _callback_counter
        _callback_counter += 1

        assert event.model is model
        assert event.model.status == ModelStatus.Training
        if event.event_type == ModelEventType.IntermediateMetric:
            _intermediates.append(event.metric)
        elif event.event_type == ModelEventType.FinalMetric:
            _finals.append(event.metric)
        elif event.event_type == ModelEventType.TrainingEnd:
            _status.append(event.status)

    engine.register_model_event_callback(ModelEventType.IntermediateMetric, callback)
    engine.register_model_event_callback(ModelEventType.FinalMetric, callback)
    engine.register_model_event_callback(ModelEventType.TrainingEnd, callback)
    model_space = MyModelSpace([1, 2, 3])
    evaluator = FunctionalEvaluator(evaluate_fn, b=5)
    exec_model_space = SimplifiedModelSpace.from_model(model_space, evaluator)

    model = exec_model_space.freeze({'a': 2})
    engine.submit_models(model)
    engine.wait_models(model)
    assert _callback_counter == 4
    assert _intermediates == [3, 4]
    assert _finals == [10]
    assert _status == [ModelStatus.Trained]
    _callback_disabled = True

    assert model.metrics.intermediates == [3, 4]
    assert model.metric == 10
    assert model.metrics.final == 10
    assert model.status == ModelStatus.Trained

    if not engine.idle_worker_available():
        time.sleep(10)  # The free event may be delayed for up to 5 seconds.
        assert engine.idle_worker_available()
    assert engine.budget_available()

    engine.submit_models(exec_model_space.freeze({'a': 3}))
    # assert engine.query_idle_workers() == 0
    assert len(list(engine.list_models())) == 2
    engine.wait_models()
    for model in engine.list_models():
        if model.status == ModelStatus.Failed:
            assert model.metrics.intermediates == [4, 5]
            assert model.metrics.final is None
            break
    else:
        assert False, 'No failed model found'
