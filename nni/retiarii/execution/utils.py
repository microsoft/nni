# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, List, cast

from nni.experiment.config.training_services import RemoteConfig
from .interface import AbstractExecutionEngine
from ..graph import Model
from ..experiment.config import (
    BaseEngineConfig, PyEngineConfig,
    CgoEngineConfig, BenchmarkEngineConfig
)

def _unpack_if_only_one(ele: List[Any]):
    if len(ele) == 1:
        return ele[0]
    return ele

def get_mutation_dict(model: Model):
    return {mut.mutator.label: _unpack_if_only_one(mut.samples) for mut in model.history}

def mutation_dict_to_summary(mutation: dict) -> dict:
    mutation_summary = {}
    for label, samples in mutation.items():
        # FIXME: this check might be wrong
        if not isinstance(samples, list):
            mutation_summary[label] = samples
        else:
            for i, sample in enumerate(samples):
                mutation_summary[f'{label}_{i}'] = sample
    return mutation_summary

def get_mutation_summary(model: Model) -> dict:
    mutation = get_mutation_dict(model)
    return mutation_dict_to_summary(mutation)

def init_execution_engine(config, port, url_prefix) -> AbstractExecutionEngine:
    if isinstance(config.execution_engine, BaseEngineConfig):
        from .base import BaseExecutionEngine
        return BaseExecutionEngine(port, url_prefix)
    elif isinstance(config.execution_engine, CgoEngineConfig):
        from .cgo_engine import CGOExecutionEngine

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
        from .python import PurePythonExecutionEngine
        return PurePythonExecutionEngine(port, url_prefix)
    elif isinstance(config.execution_engine, BenchmarkEngineConfig):
        from .benchmark import BenchmarkExecutionEngine
        assert config.execution_engine.benchmark is not None, \
            '"benchmark" must be set when benchmark execution engine is used.'
        return BenchmarkExecutionEngine(config.execution_engine.benchmark)
    else:
        raise ValueError(f'Unsupported engine type: {config.execution_engine}')