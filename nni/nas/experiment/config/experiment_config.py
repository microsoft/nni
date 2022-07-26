# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Union, Optional

from nni.experiment.config import utils, ExperimentConfig

from .engine_config import ExecutionEngineConfig

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
    execution_engine: Union[str, ExecutionEngineConfig]

    # Internal: to support customized fields in trial command
    # Useful when customized python / environment variables are needed
    _trial_command_params: Optional[Dict[str, Any]] = None

    def __init__(self, training_service_platform: Union[str, None] = None,
                 execution_engine: Union[str, ExecutionEngineConfig] = 'py',
                 **kwargs):
        super().__init__(training_service_platform, **kwargs)
        self.execution_engine = execution_engine

    def _canonicalize(self, _parents):
        msg = '{} is not supposed to be set in Retiarii experiment by users, your config is {}.'
        if self.search_space != '':
            raise ValueError(msg.format('search_space', self.search_space))
        # TODO: maybe we should also allow users to specify trial_code_directory
        if str(self.trial_code_directory) != '.' and not os.path.isabs(self.trial_code_directory):
            raise ValueError(msg.format('trial_code_directory', self.trial_code_directory))

        trial_command_tmpl = '{envs} {python} -m nni.retiarii.trial_entry {execution_engine}'
        if self.trial_command != '_reserved' and '-m nni.retiarii.trial_entry' not in self.trial_command:
            raise ValueError(msg.format('trial_command', self.trial_command))

        if isinstance(self.execution_engine, str):
            self.execution_engine = execution_engine_config_factory(self.execution_engine)

        _trial_command_params = {
            # Default variables
            'envs': '',
            # TODO: maybe use sys.executable rendered in trial side (e.g., trial_runner)
            'python': sys.executable,
            'execution_engine': self.execution_engine.name,

            # This should override the parameters above.
            **(self._trial_command_params or {})
        }

        self.trial_command = trial_command_tmpl.format(**_trial_command_params).strip()

        super()._canonicalize([self])
