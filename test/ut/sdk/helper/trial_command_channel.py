# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing_extensions import Literal
from nni.runtime.trial_command_channel import TrialCommandChannel

from nni.typehint import TrialMetric, ParameterRecord


class TestHelperTrialCommandChannel(TrialCommandChannel):

    def __init__(self):
        self._params = {
            'parameter_id': 0,
            'parameters': {}
        }
        self._last_metric = None

        self.intermediates = []
        self.final = None

    def init_params(self, params):
        self._params = copy.deepcopy(params)

    def get_last_metric(self):
        """For backward compatibility, return the last metric as the full dict."""
        return self._last_metric

    def receive_parameter(self) -> ParameterRecord | None:
        return self._params

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None,
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        self._last_metric = {
            'type': type,
            'parameter_id': parameter_id,
            'trial_job_id': trial_job_id,
            'sequence': sequence,
            'value': value
        }

        if type == 'PERIODICAL':
            self.intermediates.append(value)
        else:
            self.final = value
