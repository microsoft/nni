# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import warnings

import colorama
from typing_extensions import Literal

from nni.typehint import TrialMetric
from .base import TrialCommandChannel, ParameterRecord

_logger = logging.getLogger('nni')


class StandaloneTrialCommandChannel(TrialCommandChannel):
    """
    Special channel used when trial is running standalone,
    without tuner or NNI manager.
    """

    def receive_parameter(self) -> ParameterRecord | None:
        warning_message = ''.join([
            colorama.Style.BRIGHT,
            colorama.Fore.RED,
            'Running trial code without runtime. ',
            'Please check the tutorial if you are new to NNI: ',
            colorama.Fore.YELLOW,
            'https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html',
            colorama.Style.RESET_ALL
        ])
        warnings.warn(warning_message, RuntimeWarning)
        return ParameterRecord(
            parameter_id=None,
            parameters={}
        )

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None,  # pylint: disable=redefined-builtin
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        if type == 'FINAL':
            _logger.info('Final result: %s', value)
        elif type == 'PERIODICAL':
            _logger.info('Intermediate result: %s  (Index %s)', value, sequence)
        else:
            _logger.error('Unexpected metric type: %s', type)
