import time
import os
from typing import List

from ..graph import Model, ModelStatus
from .base import BaseExecutionEngine
from .cgo_engine import CGOExecutionEngine
from .interface import AbstractExecutionEngine, WorkerInfo
from .listener import DefaultListener

_execution_engine = None
_default_listener = None

__all__ = ['get_execution_engine', 'get_and_register_default_listener',
           'submit_models', 'wait_models', 'query_available_resources']


def get_execution_engine() -> BaseExecutionEngine:
    """
    Currently we assume the default execution engine is BaseExecutionEngine.
    """
    global _execution_engine
    if _execution_engine is None:
        if os.environ.get('CGO') == 'true':
            _execution_engine = CGOExecutionEngine()
        else:
            _execution_engine = BaseExecutionEngine()
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


def wait_models(*models: Model) -> None:
    get_and_register_default_listener(get_execution_engine())
    while True:
        time.sleep(1)
        left_models = [g for g in models if not g.status in (ModelStatus.Trained, ModelStatus.Failed)]
        if not left_models:
            break


def query_available_resources() -> List[WorkerInfo]:
    listener = get_and_register_default_listener(get_execution_engine())
    return listener.resources
