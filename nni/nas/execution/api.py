# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import warnings
from typing import Iterable, cast

from nni.experiment.config.training_services import RemoteConfig
from nni.nas.execution.common import (
    Model, ModelStatus,
    AbstractExecutionEngine,
    DefaultListener
)

_execution_engine = None
_default_listener = None

__all__ = ['init_execution_engine', 'get_execution_engine', 'get_and_register_default_listener',
           'list_models', 'submit_models', 'wait_models', 'query_available_resources',
           'set_execution_engine', 'is_stopped_exec', 'budget_exhausted']


def init_execution_engine(config, port, url_prefix) -> AbstractExecutionEngine:
    from ..experiment.config import (
        BaseEngineConfig, PyEngineConfig,
        CgoEngineConfig, BenchmarkEngineConfig
    )
    if isinstance(config.execution_engine, BaseEngineConfig):
        from .pytorch.graph import BaseExecutionEngine
        return BaseExecutionEngine(port, url_prefix)
    elif isinstance(config.execution_engine, CgoEngineConfig):
        from .pytorch.cgo.engine import CGOExecutionEngine

        assert not isinstance(config.training_service, list) \
            and config.training_service.platform == 'remote', \
            "CGO execution engine currently only supports remote training service"
        assert config.execution_engine.batch_waiting_time is not None \
            and config.execution_engine.max_concurrency_cgo is not None
        return CGOExecutionEngine(cast(RemoteConfig, config.training_service),
                                    max_concurrency=config.execution_engine.max_concurrency_cgo,
                                    batch_waiting_time=config.execution_engine.batch_waiting_time,
                                    rest_port=port,
                                    rest_url_prefix=url_prefix)
    elif isinstance(config.execution_engine, PyEngineConfig):
        from .pytorch.simplified import PurePythonExecutionEngine
        return PurePythonExecutionEngine(port, url_prefix)
    elif isinstance(config.execution_engine, BenchmarkEngineConfig):
        from .pytorch.benchmark import BenchmarkExecutionEngine
        assert config.execution_engine.benchmark is not None, \
            '"benchmark" must be set when benchmark execution engine is used.'
        return BenchmarkExecutionEngine(config.execution_engine.benchmark)
    else:
        raise ValueError(f'Unsupported engine type: {config.execution_engine}')


def set_execution_engine(engine: AbstractExecutionEngine) -> None:
    global _execution_engine
    if _execution_engine is not None:
        warnings.warn('Execution engine is already set. '
                      'You should avoid instantiating RetiariiExperiment twice in one process. '
                      'If you are running in a Jupyter notebook, please restart the kernel.',
                      RuntimeWarning)
    _execution_engine = engine


def get_execution_engine() -> AbstractExecutionEngine:
    global _execution_engine
    assert _execution_engine is not None, 'You need to set execution engine, before using it.'
    return _execution_engine


def get_and_register_default_listener(engine: AbstractExecutionEngine) -> DefaultListener:
    global _default_listener
    if _default_listener is None:
        _default_listener = DefaultListener()
        engine.register_graph_listener(_default_listener)
    return _default_listener


def submit_models(*models: Model) -> None:
    engine = get_execution_engine()
    get_and_register_default_listener(engine)
    engine.submit_models(*models)


def list_models(*models: Model) -> Iterable[Model]:
    engine = get_execution_engine()
    get_and_register_default_listener(engine)
    return engine.list_models()


def wait_models(*models: Model) -> None:
    get_and_register_default_listener(get_execution_engine())
    while True:
        time.sleep(1)
        left_models = [g for g in models if not g.status in (ModelStatus.Trained, ModelStatus.Failed)]
        if not left_models:
            break


def query_available_resources() -> int:
    engine = get_execution_engine()
    resources = engine.query_available_resource()
    return resources if isinstance(resources, int) else len(resources)


def is_stopped_exec(model: Model) -> bool:
    return model.status in (ModelStatus.Trained, ModelStatus.Failed)


def budget_exhausted() -> bool:
    engine = get_execution_engine()
    return engine.budget_exhausted()
