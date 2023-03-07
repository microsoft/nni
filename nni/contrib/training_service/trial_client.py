# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
from typing_extensions import Literal

import requests

import nni
from nni.typehint import ParameterRecord, TrialMetric
from nni.runtime.env_vars import trial_env_vars
from nni.runtime.trial_command_channel import TrialCommandChannel

from .trial_runner import TrialServerHandler
from .typehint import MetricCommand

_logger = logging.getLogger(__name__)


class TrialClient(TrialCommandChannel):
    """The client side of :class:`TrialServer`."""

    def __init__(self, url: str | None = None, trial_id: str | None = None) -> None:
        if url is not None:
            self._url = url
        else:
            self._url = TrialServerHandler.ADDRESS
        if trial_id is not None:
            self._trial_id = trial_id
        else:
            self._trial_id = trial_env_vars.NNI_TRIAL_JOB_ID

    def receive_parameter(self) -> ParameterRecord | None:
        response = requests.get(self._url + '/parameter/' + self._trial_id)
        if response.status_code != 200:
            _logger.error('Failed to receive parameter: %s', response)
            return None
        parameter = response.json()['parameter']
        if not parameter:
            _logger.warning('Received empty parameter: \'%s\'', parameter)
            return None
        if not isinstance(parameter, str):
            _logger.error('Received invalid parameter: \'%s\'', parameter)
            return None
        return nni.load(parameter)  # Unpack the parameter generated by tuner.

    def send_metric(
        self,
        type: Literal['PERIODICAL', 'FINAL'],  # pylint: disable=redefined-builtin
        parameter_id: int | None,
        trial_job_id: str,
        sequence: int,
        value: TrialMetric
    ) -> None:
        if trial_job_id != self._trial_id:
            _logger.warning('Trial job id does not match: %s vs. %s. Metric will be ignored.', trial_job_id, self._trial_id)
            return
        metric = {
            'parameter_id': parameter_id,
            'trial_job_id': trial_job_id,
            'type': type,
            'sequence': sequence,
            'value': nni.dump(value),  # Pack the metric value, which will be unpacked by tuner.
        }
        command = MetricCommand(command_type='metric', id=trial_job_id, metric=nni.dump(metric))
        response = requests.post(self._url + '/metric', json=command)
        if response.status_code != 200:
            _logger.error('Failed to send metric: %s', response)
