# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from pathlib import Path

from nni.experiment import Experiment, ExperimentConfig


class AutoCompressExperimentConfig(ExperimentConfig):
    trial_command: str = 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry'

    def __setattr__(self, key, value):
        fixed_attrs = {'trial_command': 'python3 -m nni.algorithms.compression.pytorch.auto_compress.trial_entry'}
        if key in fixed_attrs and fixed_attrs[key] != value:
            raise AttributeError(f'{key} is not supposed to be set in AutoCompress mode by users!')
        # 'trial_code_directory' is handled differently because the path will be converted to absolute path by us
        if key == 'trial_code_directory' and not (value == Path('.') or os.path.isabs(value)):
            raise AttributeError(f'{key} is not supposed to be set in Retiarii mode by users!')
        self.__dict__[key] = value

class AutoCompressExperiment(Experiment):
    pass
