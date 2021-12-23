# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for local training service.

Check the reference_ for explaination of each field.

You may also want to check `local training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _local training service doc: https://nni.readthedocs.io/en/stable/TrainingService/LocalMode.html

"""

__all__ = ['LocalConfig']

from dataclasses import dataclass
from typing import List, Optional, Union

from ..training_service import TrainingServiceConfig
from .. import utils

@dataclass(init=False)
class LocalConfig(TrainingServiceConfig):
    platform: str = 'local'
    use_active_gpu: Optional[bool] = None
    max_trial_number_per_gpu: int = 1
    gpu_indices: Union[List[int], int, str, None] = None
    reuse_mode: bool = False

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        self.gpu_indices = utils.canonical_gpu_indices(self.gpu_indices)
        self.nni_manager_ip = None

    def _validate_canonical(self):
        super()._validate_canonical()
        utils.validate_gpu_indices(self.gpu_indices)
        if self.trial_gpu_number and self.use_active_gpu is None:
            raise ValueError(
                'LocalConfig: please set use_active_gpu to True if your system has GUI, '
                'or set it to False if the computer runs multiple experiments concurrently.'
            )
        if not self.trial_gpu_number and self.max_trial_number_per_gpu != 1:
            raise ValueError('LocalConfig: max_trial_number_per_gpu does not work without trial_gpu_number')
