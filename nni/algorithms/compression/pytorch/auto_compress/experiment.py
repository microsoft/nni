# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
from pathlib import Path, PurePath
from typing import overload, Union, List

from numpy import tri

from nni.experiment import Experiment, ExperimentConfig
from nni.algorithms.compression.pytorch.auto_compress.interface import AbstractAutoCompressionModule


class AutoCompressionExperiment(Experiment):

    @overload
    def __init__(self, auto_compress_module: AbstractAutoCompressionModule, config: ExperimentConfig) -> None:
        """
        Prepare an experiment.

        Use `Experiment.run()` to launch it.

        Parameters
        ----------
        auto_compress_module
            The module provided by the user implements the `AbstractAutoCompressionModule` interfaces.
            Remember put the module file under `trial_code_directory`.
        config
            Experiment configuration.
        """
        ...

    @overload
    def __init__(self, auto_compress_module: AbstractAutoCompressionModule, training_service: Union[str, List[str]]) -> None:
        """
        Prepare an experiment, leaving configuration fields to be set later.

        Example usage::

            experiment = Experiment(auto_compress_module, 'remote')
            experiment.config.trial_command = 'python3 trial.py'
            experiment.config.machines.append(RemoteMachineConfig(ip=..., user_name=...))
            ...
            experiment.run(8080)

        Parameters
        ----------
        auto_compress_module
            The module provided by the user implements the `AbstractAutoCompressionModule` interfaces.
            Remember put the module file under `trial_code_directory`.
        training_service
            Name of training service.
            Supported value: "local", "remote", "openpai", "aml", "kubeflow", "frameworkcontroller", "adl" and hybrid training service.
        """
        ...

    def __init__(self, auto_compress_module: AbstractAutoCompressionModule, config=None, training_service=None):
        super().__init__(config, training_service)

        self.module_file_path = str(PurePath(inspect.getfile(auto_compress_module)))
        self.module_name = auto_compress_module.__name__

    def start(self, port: int, debug: bool) -> None:
        trial_code_directory = str(PurePath(Path(self.config.trial_code_directory).absolute())) + '/'
        assert self.module_file_path.startswith(trial_code_directory), 'The file path of the user-provided module should under trial_code_directory.'
        relative_module_path = self.module_file_path.split(trial_code_directory)[1]
        # only support linux, need refactor?
        command = 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry --module_file_name {} --module_class_name {}'
        self.config.trial_command = command.format(relative_module_path, self.module_name)
        super().start(port=port, debug=debug)
