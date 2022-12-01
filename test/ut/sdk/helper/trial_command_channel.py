# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing_extensions import Literal
from nni.runtime.trial_command_channel import TrialCommandChannel

from nni import dump
from nni.typehint import TrialMetric, ParameterRecord


class TestHelperTrialCommandChannel(TrialCommandChannel):

    def init_params(self, params):
        self._params = copy.deepcopy(params)

    def get_last_metric(self):
        return self._last_metric

    def receive_parameter(self) -> ParameterRecord | None:
        return self._params

    def send_metric(self, type: Literal['INTERMEDIATE', 'FINAL'], parameter_id: int | None,
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        self._last_metric = {
            'type': type,
            'parameter_id': parameter_id,
            'trial_job_id': trial_job_id,
            'sequence': sequence,
            'value': value
        }
