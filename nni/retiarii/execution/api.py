# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import Iterable

from ..graph import Model, ModelStatus
from .interface import AbstractExecutionEngine
from .listener import DefaultListener

_execution_engine = None
_default_listener = None

__all__ = ['get_execution_engine', 'get_and_register_default_listener',
           'list_models', 'submit_models', 'wait_models', 'query_available_resources',
           'set_execution_engine', 'is_stopped_exec', 'budget_exhausted']


def set_execution_engine(engine: AbstractExecutionEngine) -> None:
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = engine
    else:
        raise RuntimeError('Execution engine is already set.')


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
