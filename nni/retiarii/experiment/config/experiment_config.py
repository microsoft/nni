# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from dataclasses import dataclass
from typing import Any, Optional, Union

from nni.experiment.config import utils, ExperimentConfig

from .engine_config import ExecutionEngineConfig, PyEngineConfig

__all__ = ['RetiariiExeConfig']

def execution_engine_config_factory(engine_name):
    # FIXME: may move this function to experiment utils in future
    cls = _get_ee_config_class(engine_name)
    if cls is None:
        raise ValueError(f'Invalid execution engine name: {engine_name}')
    return cls()

def _get_ee_config_class(engine_name):
    for cls in ExecutionEngineConfig.__subclasses__():
        if cls.name == engine_name:
            return cls
    return None

@dataclass(init=False)
class RetiariiExeConfig(ExperimentConfig):
    # FIXME: refactor this class to inherit from a new common base class with HPO config
    search_space: Any = ''
    trial_code_directory: utils.PathLike = '.'
    trial_command: str = '_reserved'
    # new config field for NAS
    execution_engine: ExecutionEngineConfig = PyEngineConfig()

    def __init__(self, training_service_platform: Optional[str] = None,
                 execution_engine: Union[str, ExecutionEngineConfig] = None, #TODO: having default value or not?
                 **kwargs):
        super().__init__(training_service_platform, **kwargs)

        if execution_engine is not None:
            # the user chose to init with `config = ExperimentConfig('local')` and set fields later
            # we need to create empty training service & algorithm configs to support `config.tuner.name = 'random'`
            assert utils.is_missing(self.execution_engine)
            if isinstance(execution_engine, str):
                self.execution_engine = execution_engine_config_factory(execution_engine)
            else:
                self.execution_engine = execution_engine

        self.__dict__['trial_command'] = 'python3 -m nni.retiarii.trial_entry ' + self.execution_engine.name

    def __setattr__(self, key, value):
        #TODO: tuner settings can also be blocked here
        fixed_attrs = {'search_space': '',
                       'trial_command': '_reserved'}
        if key in fixed_attrs and fixed_attrs[key] != value:
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (str(value) == '.' or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        #if key == 'execution_engine':
        #    assert value in ['base', 'py', 'cgo', 'benchmark', 'oneshot'], f'The specified execution engine "{value}" is not supported.'
        #    self.__dict__['trial_command'] = 'python3 -m nni.retiarii.trial_entry ' + value
        super().__setattr__(key, value) #TODO: double check whether new fields are validated

    def _validate_canonical(self):
        super()._validate_canonical(False)