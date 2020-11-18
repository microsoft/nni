import time
import importlib.util
from typing import *

from ..graph import Model, ModelStatus
from .base import BaseExecutionEngine
from .interface import *
from .listener import DefaultListener

_execution_engine = None
_default_listener = None

__all__ = ['get_execution_engine', 'get_and_register_default_listener',
           'submit_models', 'wait_models', 'query_available_resources',
           'get_base_model_ir', 'get_specified_mutators', 'get_trainer']


def get_execution_engine() -> BaseExecutionEngine:
    """
    Currently we assume the default execution engine is BaseExecutionEngine.
    """
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = BaseExecutionEngine()
    return _execution_engine


def get_and_register_default_listener(engine: AbstractExecutionEngine) -> DefaultListener:
    global _default_listener
    if _default_listener is None:
        _default_listener = DefaultListener()
        engine.register_graph_listener(_default_listener)
    return _default_listener

def _get_search_space() -> 'Dict':
    engine = get_execution_engine()
    while True:
        time.sleep(1)
        if engine.get_search_space() is not None:
            break
    return engine.get_search_space()

def get_base_model_ir() -> 'Model':
    search_space = _get_search_space()
    return Model._load(search_space['base_model_ir'])

def get_specified_mutators() -> List['Mutator']:
    search_space = _get_search_space()
    applied_mutators = []
    for each in search_space['applied_mutators']:
        spec = importlib.util.spec_from_file_location("module.name", each['filepath'])
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        #m.BlockMutator()
        class_constructor = getattr(m, each['classname'])
        mutator = class_constructor(**each['args'])
        applied_mutators.append(mutator)
    return applied_mutators

def get_trainer() -> 'BaseTrainer':
    search_space = _get_search_space()
    return search_space['training_approach']

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
