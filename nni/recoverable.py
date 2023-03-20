# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import nni

class Recoverable:
    def __init__(self):
        self.recovered_max_param_id = -1
        self.recovered_trial_params = {}

    def load_checkpoint(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def get_checkpoint_path(self) -> str | None:
        ckp_path = os.getenv('NNI_CHECKPOINT_DIRECTORY')
        if ckp_path is not None and os.path.isdir(ckp_path):
            return ckp_path
        return None

    def recover_parameter_id(self, data) -> int:
        # this is for handling the resuming of the interrupted data: parameters
        if not isinstance(data, list):
            data = [data]

        previous_max_param_id = 0
        for trial in data:
            # {'parameter_id': 0, 'parameter_source': 'resumed', 'parameters': {'batch_size': 128, ...}
            if isinstance(trial, str):
                trial = nni.load(trial)
            if not isinstance(trial['parameter_id'], int):
                # for dealing with user customized trials
                # skip for now
                continue
            self.recovered_trial_params[trial['parameter_id']] = trial['parameters']
            if previous_max_param_id < trial['parameter_id']:
                previous_max_param_id = trial['parameter_id']
        self.recovered_max_param_id = previous_max_param_id
        return previous_max_param_id

    def is_created_in_previous_exp(self, param_id: int | None) -> bool:
        if param_id is None:
            return False
        return param_id <= self.recovered_max_param_id

    def get_previous_param(self, param_id: int) -> dict:
        return self.recovered_trial_params[param_id]