# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import dataclasses
from pathlib import Path
from subprocess import Popen
from typing import Optional

import nni
from nni.experiment import Experiment, ExperimentConfig, AlgorithmConfig


class AutoCompressExperimentConfig(ExperimentConfig):
    auto_compress_module_file_name: str = './auto_compress_module.py'

    def __setattr__(self, key, value):
        fixed_attrs = {'trial_command': 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry'}
        if key in fixed_attrs and type(value) is not type(dataclasses.MISSING) and not value.startswith(fixed_attrs[key]):
            raise AttributeError(f'{key} is not supposed to be set in AutoCompress mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (value == Path('.') or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in AutoCompress mode by users!')
        self.__dict__[key] = value

    def validate(self, initialized_tuner: bool = False) -> None:
        super().validate(initialized_tuner=initialized_tuner)
        if not Path(self.trial_code_directory, self.auto_compress_module_file_name).exists():
            raise ValueError('{} not exsisted under {}'.format(self.auto_compress_module_file_name, self.trial_code_directory))

class AutoCompressExperiment(Experiment):
    def __init__(self, config=None, training_service=None):
        nni.runtime.log.init_logger_experiment()

        self.config: Optional[AutoCompressExperimentConfig] = None
        self.id: Optional[str] = None
        self.port: Optional[int] = None
        self._proc: Optional[Popen] = None
        self.mode = 'new'

        args = [config, training_service]  # deal with overloading
        if isinstance(args[0], (str, list)):
            self.config = AutoCompressExperimentConfig(args[0])
            self.config.tuner = AlgorithmConfig(name='_none_', class_args={})
            self.config.assessor = AlgorithmConfig(name='_none_', class_args={})
            self.config.advisor = AlgorithmConfig(name='_none_', class_args={})
        else:
            self.config = args[0]

    def start(self, port: int, debug: bool) -> None:
        self.config.trial_command = 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry --module_file_name {}'.format(self.config.auto_compress_module_file_name)
        super().start(port=port, debug=debug)
