# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import cast
from typing_extensions import Literal

from nni.common import dump, load
from nni.typehint import TrialMetric
from .base import TrialCommandChannel, ParameterRecord
from ..env_vars import trial_env_vars


class LocalLegacyTrialCommandChannel(TrialCommandChannel):
    """
    Command channel based on a local file system.
    This is the legacy implementation before NNI v3.0.
    """

    def __init__(self):
        self._sysdir = trial_env_vars.NNI_SYS_DIR
        if not os.path.exists(os.path.join(self._sysdir, '.nni')):
            os.makedirs(os.path.join(self._sysdir, '.nni'))
        self._metric_file = open(os.path.join(self._sysdir, '.nni', 'metrics'), 'ab')

        self._outputdir = trial_env_vars.NNI_OUTPUT_DIR
        if not os.path.exists(self._outputdir):
            os.makedirs(self._outputdir)

        self._reuse_mode = trial_env_vars.REUSE_MODE
        self._nni_platform = trial_env_vars.NNI_PLATFORM

        self._multiphase = trial_env_vars.MULTI_PHASE

        self._param_index = 0

    def _send(self, string) -> None:
        if self._nni_platform != 'local' or self._reuse_mode in ('true', 'True'):
            assert len(string) < 1000000, 'Metric too long'
            print("NNISDK_MEb'%s'" % (string), flush=True)
        else:
            data = (string + '\n').encode('utf8')
            assert len(data) < 1000000, 'Metric too long'
            self._metric_file.write(b'ME%06d%b' % (len(data), data))
            self._metric_file.flush()
            if sys.platform == "win32":
                file = open(self._metric_file.name)
                file.close()
            else:
                subprocess.run(['touch', self._metric_file.name], check=True)

    def _request_next_parameter(self) -> None:
        metric = dump({
            'trial_job_id': trial_env_vars.NNI_TRIAL_JOB_ID,  # TODO: shouldn't rely on env vars
            'type': 'REQUEST_PARAMETER',
            'sequence': 0,
            'parameter_index': self._param_index
        })
        self._send(metric)

    def receive_parameter(self) -> ParameterRecord | None:
        params_file_name = ''
        if self._multiphase in ('true', 'True'):
            params_file_name = ('parameter_{}.cfg'.format(self._param_index), 'parameter.cfg')[self._param_index == 0]
        else:
            if self._param_index > 0:
                return None
            elif self._param_index == 0:
                params_file_name = 'parameter.cfg'
            else:
                raise AssertionError('self._param_index value ({}) should >=0'.format(self._param_index))

        params_filepath = os.path.join(self._sysdir, params_file_name)
        if not os.path.isfile(params_filepath):
            self._request_next_parameter()
        while not (os.path.isfile(params_filepath) and os.path.getsize(params_filepath) > 0):
            time.sleep(3)
        params_file = open(params_filepath, 'r')
        params = load(fp=params_file)
        self._param_index += 1
        assert isinstance(params, dict) and 'parameters' in params
        return cast(ParameterRecord, params)

    def send_metric(self, type: Literal['PERIODICAL', 'FINAL'], parameter_id: int | None,  # pylint: disable=redefined-builtin
                    trial_job_id: str, sequence: int, value: TrialMetric) -> None:
        dumped_metric = dump({
            'parameter_id': parameter_id,
            'trial_job_id': trial_job_id,
            'type': type,
            'sequence': sequence,
            'value': dump(value)
        })
        self._send(dumped_metric)
