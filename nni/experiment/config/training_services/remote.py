# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for remote training service.

Check the reference_ for explaination of each field.

You may also want to check `remote training service doc`_.

.. _reference: https://nni.readthedocs.io/en/stable/reference/experiment_config.html

.. _remote training service doc: https://nni.readthedocs.io/en/stable/TrainingService/RemoteMachineMode.html

"""

__all__ = ['RemoteConfig', 'RemoteMachineConfig']

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import warnings

from typing_extensions import Literal

from ..base import ConfigBase
from ..training_service import TrainingServiceConfig
from .. import utils

@dataclass(init=False)
class RemoteMachineConfig(ConfigBase):
    host: str
    port: int = 22
    user: str
    password: Optional[str] = None
    ssh_key_file: Optional[utils.PathLike] = '~/.ssh/id_rsa'
    ssh_passphrase: Optional[str] = None
    use_active_gpu: bool = False
    max_trial_number_per_gpu: int = 1
    gpu_indices: Union[List[int], int, str, None] = None
    python_path: Optional[str] = None

    def _canonicalize(self, parents):
        super()._canonicalize(parents)
        if self.password is not None:
            self.ssh_key_file = None
        self.gpu_indices = utils.canonical_gpu_indices(self.gpu_indices)

    def _validate_canonical(self):
        super()._validate_canonical()

        assert 0 < self.port < 65536
        assert self.max_trial_number_per_gpu > 0
        utils.validate_gpu_indices(self.gpu_indices)

        if self.password is not None:
            warnings.warn('SSH password will be exposed in web UI as plain text. We recommend to use SSH key file.')
        elif not Path(self.ssh_key_file).is_file():  # type: ignore
            raise ValueError(
                f'RemoteMachineConfig: You must either provide password or a valid SSH key file "{self.ssh_key_file}"'
            )

@dataclass(init=False)
class RemoteConfig(TrainingServiceConfig):
    platform: Literal['remote'] = 'remote'
    machine_list: List[RemoteMachineConfig]
    reuse_mode: bool = False
    #log_collection: Literal['on_error', 'always', 'never'] = 'on_error'  # TODO: NNI_OUTPUT_DIR?

    def _validate_canonical(self):
        super()._validate_canonical()
        if not self.machine_list:
            raise ValueError(f'RemoteConfig: must provide at least one machine in machine_list')
        if not self.trial_gpu_number and any(machine.max_trial_number_per_gpu != 1 for machine in self.machine_list):
            raise ValueError('RemoteConfig: max_trial_number_per_gpu does not work without trial_gpu_number')
        #if self.reuse_mode and self.log_collection != 'on_error':
        #    raise ValueError('RemoteConfig: log_collection is not supported in reuse mode')
