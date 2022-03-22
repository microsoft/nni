# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import inspect
from pathlib import Path, PurePath

from nni.experiment import Experiment, ExperimentConfig
from nni.algorithms.compression.pytorch.auto_compress.interface import AbstractAutoCompressionModule


class AutoCompressionExperiment(Experiment):

    def __init__(self, auto_compress_module: AbstractAutoCompressionModule, config_or_platform: ExperimentConfig | str | list[str]) -> None:
        """
        Prepare an auto compression experiment.

        Parameters
        ----------
        auto_compress_module
            The module provided by the user implements the `AbstractAutoCompressionModule` interfaces.
            Remember put the module file under `trial_code_directory`.
        config_or_platform
            Experiment configuration or training service name.
        """
        super().__init__(config_or_platform)

        self.module_file_path = str(PurePath(inspect.getfile(auto_compress_module)))
        self.module_name = auto_compress_module.__name__

    def start(self, port: int, debug: bool) -> None:
        trial_code_directory = str(PurePath(Path(self.config.trial_code_directory).absolute())) + '/'
        assert self.module_file_path.startswith(trial_code_directory), \
            'The file path of the user-provided module should under trial_code_directory.'
        relative_module_path = self.module_file_path.split(trial_code_directory)[1]
        # only support linux, need refactor?
        command = 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry --module_file_name {} --module_class_name {}'
        self.config.trial_command = command.format(relative_module_path, self.module_name)
        super().start(port=port, debug=debug)
